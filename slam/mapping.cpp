#include "slam/mapping.hpp"

#include <iostream>
#include <utility>

slam::Mapping::UpdateStats slam::Mapping::update(TrackManager &trackManager, FrameId currentFrameId,
                                                 const geometry::CameraPose &relativePose,
                                                 const CameraPose &previousWorldPose,
                                                 const cv::Mat &cameraMatrix)
{
    UpdateStats stats;
    bool mapChanged = false;

    if (relativePose.R.empty() || relativePose.t.empty() ||
        previousWorldPose.Rwc.empty() || previousWorldPose.twc.empty() ||
        cameraMatrix.empty())
    {
        return stats;
    }

    cameraMatrix_ = cameraMatrix;

    const auto currentKeyframeId = map_.keyframeForFrame(currentFrameId);

    // Phase 1: tracks already linked to a MapPoint just pick up this frame's
    // observation -- the landmark already exists, so this is a plain
    // observation update, not a re-triangulation.
    if (currentKeyframeId)
    {
        for (const auto &[trackId, track] : trackManager.tracks())
        {
            if (!track.active() || !track.mapPointId() ||
                track.lastObservation().frameId != currentFrameId)
            {
                continue;
            }

            const auto &observation = track.lastObservation();
            if (map_.addObservation(*track.mapPointId(),
                                    MapObservation{*currentKeyframeId, observation.keypointIndex, observation.pixel}))
            {
                mapChanged = true;
            }
        }
    }

    // Phase 2: tracks with enough baseline but no MapPoint yet get
    // triangulated and promoted for the first time.
    const geometry::CameraPose identityPose{cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F)};

    for (const auto &[trackId, track] : trackManager.tracks())
    {
        if (!track.active() || track.mapPointId() ||
            track.observations().size() < 4 ||
            track.lastObservation().frameId != currentFrameId)
        {
            continue;
        }

        // TODO:
        // Don't always triangulate using the last two observations.
        // Choose the observation pair with the largest baseline/parallax
        // once KeyFrames are introduced.
        const auto &previousObservation = track.observations()[track.observations().size() - 2];
        const auto &currentObservation = track.lastObservation();

        const auto worldPoint = triangulateWorldPoint(
            cv::Point2d(previousObservation.pixel), cv::Point2d(currentObservation.pixel),
            identityPose, relativePose, cameraMatrix, previousWorldPose);
        if (!worldPoint)
        {
            continue;
        }

        const MapPointId mapPointId = map_.createPoint(*worldPoint, track.observations().size());
        trackManager.linkTrackToMapPoint(trackId, mapPointId);

        // Only observations whose source Frame was promoted to a KeyFrame
        // become part of the MapPoint's permanent observation set -- the
        // rest were tracking-only frames and leave no trace here.
        std::vector<KeyframeId> keyframesObserved;
        for (const auto &observation : track.observations())
        {
            const auto keyframeId = map_.keyframeForFrame(observation.frameId);
            if (!keyframeId)
            {
                continue;
            }

            map_.addObservation(mapPointId,
                                 MapObservation{*keyframeId, observation.keypointIndex, observation.pixel});
            keyframesObserved.push_back(*keyframeId);
        }

        ++stats.newMapPoints;
        stats.promotions.emplace_back(trackId, mapPointId);
        mapChanged = true;
    }

    if (mapChanged)
    {
        covisibilityGraph_.rebuild(map_);
    }

    return stats;
}

std::optional<cv::Point3d> slam::Mapping::triangulateWorldPoint(
    const cv::Point2d &pixelPrev, const cv::Point2d &pixelCurr,
    const geometry::CameraPose &pose1, const geometry::CameraPose &pose2,
    const cv::Mat &cameraMatrix, const CameraPose &previousWorldPose)
{
    const cv::Point3d pointCamera = triangulator_.triangulatePoint(pixelPrev, pixelCurr, pose1, pose2, cameraMatrix);
    if (pointCamera.z <= 0.0)
    {
        return std::nullopt;
    }

    const cv::Mat Xc = (cv::Mat_<double>(3, 1) << pointCamera.x, pointCamera.y, pointCamera.z);
    const cv::Mat Xw = previousWorldPose.Rwc * Xc + previousWorldPose.twc;

    return cv::Point3d{Xw.at<double>(0), Xw.at<double>(1), Xw.at<double>(2)};
}

bool slam::Mapping::shouldCreateKeyframe(const CameraPose &currentPose) const
{
    if (!lastKeyframePose_ || currentPose.Rwc.empty() || currentPose.twc.empty())
    {
        return true;
    }

    const double translation = cv::norm(currentPose.twc - lastKeyframePose_->twc);

    const cv::Mat relativeRotation = lastKeyframePose_->Rwc.t() * currentPose.Rwc;
    cv::Mat rotationVector;
    cv::Rodrigues(relativeRotation, rotationVector);
    const double rotationDeg = cv::norm(rotationVector) * 180.0 / CV_PI;

    return translation > keyframeTranslationThreshold_ || rotationDeg > keyframeRotationThresholdDeg_;
}

slam::KeyframeId slam::Mapping::createKeyframe(FrameId sourceFrameId, const CameraPose &pose,
                                                std::vector<vision::KeyPoint> keypoints,
                                                std::vector<vision::OrbDescriptor> descriptors)
{
    lastKeyframePose_ = pose;
    KeyframeId id = map_.createKeyframe(sourceFrameId, pose, std::move(keypoints), std::move(descriptors));
    covisibilityGraph_.rebuild(map_);
    return id;
}

const slam::Map &slam::Mapping::map() const
{
    return map_;
}

slam::Map &slam::Mapping::map()
{
    return map_;
}

const slam::CovisibilityGraph &slam::Mapping::covisibilityGraph() const
{
    return covisibilityGraph_;
}

const cv::Mat &slam::Mapping::cameraMatrix() const
{
    return cameraMatrix_;
}

void slam::Mapping::refreshCovisibilityGraph()
{
    covisibilityGraph_.rebuild(map_);
}
