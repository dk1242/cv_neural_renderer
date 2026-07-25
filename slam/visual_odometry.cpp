#include "visual_odometry.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>

namespace
{
    cv::Mat buildCameraMatrix(int width, int height)
    {
        const double focalLength = 800.0;
        const double cx = width * 0.5;
        const double cy = height * 0.5;

        return (cv::Mat_<double>(3, 3) << focalLength, 0.0, cx,
                0.0, focalLength, cy,
                0.0, 0.0, 1.0);
    }
}

slam::VisualOdometry::VisualOdometry() : localMapping_(mapping_)
{
}

slam::Frame slam::VisualOdometry::createFrame(const cv::Mat &image)
{
    Frame frame;
    if (image.channels() == 3)
    {
        cv::cvtColor(image, frame.image, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, frame.image, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        frame.image = image.clone();
    }

    frame.keypoints = fastDetector_.detect(frame.image);
    frame.keypoints =
        fastDetector_.selectKeypointsInGrid(frame.image, frame.keypoints, 8, 8, 300);
    orbExtractor_.computeOrientations(frame.image, frame.keypoints);
    frame.descriptors = orbExtractor_.computeDescriptors(frame.image, frame.keypoints);
    return frame;
}

void slam::VisualOdometry::setPreviousFrame(Frame frame)
{
    previousFrame_ = std::move(frame);
}

slam::Observation slam::VisualOdometry::makeObservation(FrameId frameId, size_t keypointIndex,
                                                          const Frame &frame) const
{
    return Observation{
        frameId,
        keypointIndex,
        frame.keypoints[keypointIndex].position,
        frame.descriptors[keypointIndex]};
}

void slam::VisualOdometry::logFrameStats(size_t matchCount, std::optional<size_t> inlierCount,
                                          const TrackManager::UpdateStats &trackStats,
                                          const std::string &statusLine, double elapsedMs) const
{
    const auto lengthStats = trackManager_.activeTrackLengthStats();

    std::cout << "Frame " << frameIndex_ << " -> " << (frameIndex_ + 1) << "\n"
              << "----------------------------------------\n"
              << "Matches           : " << matchCount << "\n"
              << "RANSAC Inliers    : ";
    if (inlierCount)
    {
        std::cout << *inlierCount;
    }
    else
    {
        std::cout << "-";
    }
    std::cout << "\n\n"
              << "Tracks\n"
              << "  New             : " << trackStats.newTracks << "\n"
              << "  Extended        : " << trackStats.extendedTracks << "\n"
              << "  Terminated      : " << trackStats.terminatedTracks << "\n\n"
              << "Active            : " << trackManager_.activeTrackCount() << "\n"
              << "Total             : " << trackManager_.totalTrackCount() << "\n\n"
              << "Track Lengths\n"
              << "  Average         : " << std::fixed << std::setprecision(1) << lengthStats.average << "\n"
              << "  Longest         : " << lengthStats.longest << "\n\n";

    if (!statusLine.empty())
    {
        std::cout << statusLine << "\n";
    }

    std::cout << "Time              : " << std::fixed << std::setprecision(1)
              << elapsedMs << " ms" << std::endl;
}

void slam::VisualOdometry::logMappingStats(const Mapping::UpdateStats &mappingStats) const
{
    std::cout << "Frame " << frameIndex_ << "\n\n"
              << "Tracks\n"
              << "------\n"
              << "Active:              " << trackManager_.activeTrackCount() << "\n"
              << "Total:               " << trackManager_.totalTrackCount() << "\n\n"
              << "Map\n"
              << "---\n"
              << "New MapPoints:       " << mappingStats.newMapPoints << "\n"
              << "Total MapPoints:     " << mapping_.map().size() << "\n";

    for (const auto &[trackId, mapPointId] : mappingStats.promotions)
    {
        std::cout << "Track " << trackId << " -> MapPoint " << mapPointId << '\n';
    }

    std::cout << std::endl;
}

void slam::VisualOdometry::processFrame(const cv::Mat &frame)
{
    if (frame.empty())
    {
        std::cerr << "VisualOdometry: received an empty frame" << std::endl;
        return;
    }

    Frame currentFrame = createFrame(frame);

    if (cameraMatrix_.empty())
    {
        cameraMatrix_ = buildCameraMatrix(currentFrame.image.cols, currentFrame.image.rows);
    }

    if (!initialized_)
    {
        initialized_ = true;
        setPreviousFrame(std::move(currentFrame));
        currentPose_.Rwc = cv::Mat::eye(3, 3, CV_64F);
        currentPose_.twc = cv::Mat::zeros(3, 1, CV_64F);
        trajectory_.push_back(currentPose_);

        lastRelativePose_.reset();
        std::cout << "Loaded first frame with "
                  << previousFrame_.keypoints.size() << " keypoints" << std::endl;

        if (mapping_.shouldCreateKeyframe(currentPose_))
        {
            const KeyframeId keyframeId = mapping_.createKeyframe(
                frameIndex_, currentPose_, previousFrame_.keypoints, previousFrame_.descriptors);
            localMapping_.insertKeyframe(keyframeId);
            localMapping_.process();
            std::cout << "Frame " << frameIndex_ << " -> Created KF" << keyframeId << std::endl;
        }

        ++frameIndex_;
        return;
    }

    const auto knnForward =
        matcher_.knnMatch(previousFrame_.descriptors, currentFrame.descriptors, 2);
    const auto ratioMatches = matcher_.ratioTest(knnForward, 0.75f);
    const auto knnBackward =
        matcher_.knnMatch(currentFrame.descriptors, previousFrame_.descriptors, 2);
    const auto ratioBackwardMatches = matcher_.ratioTest(knnBackward, 0.75f);
    const auto crossMatches =
        matcher_.crosscheckMatch(ratioMatches, ratioBackwardMatches);

    if (crossMatches.size() < 30)
    {
        trackManager_.update({});

        // logMappingStats(Mapping::UpdateStats{});

        setPreviousFrame(std::move(currentFrame));
        lastRelativePose_.reset();
        ++frameIndex_;
        return;
    }

    std::vector<cv::Point2d> points1;
    std::vector<cv::Point2d> points2;
    points1.reserve(crossMatches.size());
    points2.reserve(crossMatches.size());
    for (const auto &match : crossMatches)
    {
        const auto &previousKeypoint = previousFrame_.keypoints[match.queryIdx];
        const auto &currentKeypoint = currentFrame.keypoints[match.trainIdx];
        points1.emplace_back(previousKeypoint.position.x, previousKeypoint.position.y);
        points2.emplace_back(currentKeypoint.position.x, currentKeypoint.position.y);
    }

    const auto ransacResult =
        fundamentalRansac_.estimateFundamental(points1, points2);
    const cv::Mat &fundamentalMatrix = ransacResult.model;

    std::vector<cv::Point2d> inlierPoints1;
    std::vector<cv::Point2d> inlierPoints2;
    std::vector<slam::Correspondence> correspondences;
    inlierPoints1.reserve(ransacResult.inlierCount);
    inlierPoints2.reserve(ransacResult.inlierCount);
    correspondences.reserve(ransacResult.inlierCount);
    for (int i = 0; i < ransacResult.inlierMask.rows; ++i)
    {
        if (ransacResult.inlierMask.at<uchar>(i))
        {
            inlierPoints1.push_back(points1[static_cast<size_t>(i)]);
            inlierPoints2.push_back(points2[static_cast<size_t>(i)]);

            const auto &match = crossMatches[static_cast<size_t>(i)];
            correspondences.push_back(Correspondence{
                makeObservation(frameIndex_ - 1, static_cast<size_t>(match.queryIdx), previousFrame_),
                makeObservation(frameIndex_, static_cast<size_t>(match.trainIdx), currentFrame)});
        }
    }

    trackManager_.update(correspondences);

    Mapping::UpdateStats mappingStats;
    bool validPose = true;
    int positiveDepthCount = 0;
    double rotationAngleDeg = 0.0;
    cv::Vec3d translationDirection(0.0, 0.0, 0.0);
    std::string failureReason;
    geometry::CameraPose recoveredPose;

    if (fundamentalMatrix.empty() || fundamentalMatrix.rows != 3 || fundamentalMatrix.cols != 3)
    {
        validPose = false;
        failureReason = "invalid fundamental matrix";
    }
    else if (ransacResult.inlierCount < 20)
    {
        validPose = false;
        failureReason = "insufficient inliers";
    }
    else
    {
        cv::Mat essentialMatrix =
            essentialEstimator_.computeEssentialMatrix(fundamentalMatrix, cameraMatrix_);
        essentialMatrix =
            essentialEstimator_.enforceEssentialConstraints(essentialMatrix);

        if (essentialMatrix.empty())
        {
            validPose = false;
            failureReason = "invalid essential matrix";
        }
        else
        {
            recoveredPose = poseRecovery_.recoverPose(essentialMatrix, inlierPoints1, inlierPoints2, cameraMatrix_);

            if (recoveredPose.R.empty() || recoveredPose.t.empty())
            {
                validPose = false;
                failureReason = "pose recovery failed";
            }
            else
            {
                std::optional<KeyframeId> createdKeyframeId;
                positiveDepthCount = poseRecovery_.countPositiveDepth(recoveredPose, inlierPoints1, inlierPoints2, cameraMatrix_);

                const CameraPose previousWorldPoseForMapping = previousPose_;
                previousPose_ = currentPose_;

                cv::Mat R12 = recoveredPose.R.t();
                cv::Mat t12 = -R12 * recoveredPose.t;

                if (currentPose_.Rwc.empty() || currentPose_.twc.empty())
                {
                    validPose = false;
                    failureReason = "current pose uninitialized";
                }
                else
                {
                    cv::Mat Rwc_next = currentPose_.Rwc * R12;
                    cv::Mat twc_next = currentPose_.twc + currentPose_.Rwc * t12;

                    currentPose_.Rwc = Rwc_next;
                    currentPose_.twc = twc_next;
                    trajectory_.push_back(currentPose_);

                    cv::Mat rotationVector;
                    cv::Rodrigues(currentPose_.Rwc, rotationVector);
                    rotationAngleDeg = cv::norm(rotationVector) * 180.0 / CV_PI;

                    const double translationNorm = cv::norm(recoveredPose.t);
                    if (translationNorm > std::numeric_limits<double>::epsilon())
                    {
                        translationDirection = cv::Vec3d(recoveredPose.t.at<double>(0),
                                                         recoveredPose.t.at<double>(1),
                                                         recoveredPose.t.at<double>(2)) /
                                               translationNorm;
                    }

                    lastRelativePose_ = RelativePose{recoveredPose.R.clone(), recoveredPose.t.clone(), ransacResult.inlierCount};

                    if (mapping_.shouldCreateKeyframe(currentPose_))
                    {
                        const KeyframeId keyframeId = mapping_.createKeyframe(
                            frameIndex_, currentPose_, currentFrame.keypoints, currentFrame.descriptors);
                        createdKeyframeId = keyframeId;
                        std::cout << "Frame " << frameIndex_ << " -> Created KF" << keyframeId << std::endl;
                    }
                }

                // Runs after this frame's KeyFrame (if any) has been created,
                // so newly promoted MapPoints can pick up an observation on
                // frameIndex_ itself via Map::keyframeForFrame instead of
                // always missing the frame that triggered the promotion.
                mappingStats = mapping_.update(trackManager_, frameIndex_, recoveredPose, previousWorldPoseForMapping, cameraMatrix_);
                if(createdKeyframeId)
                {
                    localMapping_.insertKeyframe(*createdKeyframeId);
                    localMapping_.process();
                }
            }
        }
    }

    if (!validPose)
    {
        lastRelativePose_.reset();
    }

    // logMappingStats(mappingStats);

    setPreviousFrame(std::move(currentFrame));
    ++frameIndex_;
}

const std::optional<slam::RelativePose> &slam::VisualOdometry::lastRelativePose() const
{
    return lastRelativePose_;
}

void slam::VisualOdometry::setCameraCenters()
{
    for (const auto &pose : trajectory_)
    {
        cameraCenters_.emplace_back(
            pose.twc.at<double>(0),
            pose.twc.at<double>(1),
            pose.twc.at<double>(2));
    }
}

const slam::Map& slam::VisualOdometry::map() const
{
    return mapping_.map();
}

const std::vector<cv::Point3d>& slam::VisualOdometry::cameraCenters() const
{
    return cameraCenters_;
}

const slam::CameraPose& slam::VisualOdometry::currentPose() const
{
    return currentPose_;
}

const slam::TrackManager& slam::VisualOdometry::trackManager() const
{
    return trackManager_;
}

cv::Mat slam::VisualOdometry::drawTrajectory() const
{
    // double minX = std::numeric_limits<double>::max();
    // double maxX = std::numeric_limits<double>::lowest();

    // double minY = std::numeric_limits<double>::max();
    // double maxY = std::numeric_limits<double>::lowest();

    // double minZ = std::numeric_limits<double>::max();
    // double maxZ = std::numeric_limits<double>::lowest();

    // for (const auto &point : mapPoints_)
    // {
    //     minX = std::min(minX, point.position.x);
    //     maxX = std::max(maxX, point.position.x);

    //     minY = std::min(minY, point.position.y);
    //     maxY = std::max(maxY, point.position.y);

    //     minZ = std::min(minZ, point.position.z);
    //     maxZ = std::max(maxZ, point.position.z);
    // }
    // std::cout
    //     << "X: [" << minX << ", " << maxX << "]\n"
    //     << "Y: [" << minY << ", " << maxY << "]\n"
    //     << "Z: [" << minZ << ", " << maxZ << "]\n";
    cv::Mat canvas(1000, 1000, CV_8UC3, cv::Scalar(255, 255, 255));

    const int centerX = canvas.cols / 2;
    const int centerY = canvas.rows / 2;

    const double scale = 30.0;

    for (size_t i = 1; i < cameraCenters_.size(); ++i)
    {
        const auto &p0 = cameraCenters_[i - 1];
        const auto &p1 = cameraCenters_[i];

        cv::Point pt0(
            centerX + static_cast<int>(p0.x * scale),
            centerY - static_cast<int>(p0.z * scale));

        cv::Point pt1(
            centerX + static_cast<int>(p1.x * scale),
            centerY - static_cast<int>(p1.z * scale));
        
        cv::line(canvas, pt0, pt1, cv::Scalar(0, 0, 255), 2);
    }
    for (const auto &[id, point] : mapping_.map().points())
    {
        cv::Point pt(centerX + point.position.x * scale,
                     centerY - point.position.z * scale);

        cv::circle(canvas, pt, 3, cv::Scalar(255, 0, 0), -1);
    }
    return canvas;
}