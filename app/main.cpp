#include <iostream>
#include <iomanip>
#include <chrono>
#include <filesystem>
#include <limits>

#include <opencv2/opencv.hpp>

#include "vision/image_pyramid.h"
#include "vision/orb.hpp"
#include "vision/fast.hpp"
#include "vision/descriptor_matcher.h"
#include "vision/visualizer.h"

#include "geometry/ransac.hpp"
#include "geometry/homography.hpp"
#include "geometry/epipolar_geometry.hpp"
#include "geometry/fundamental_matrix.h"
#include "geometry/normalization.h"
#include "geometry/essential_matrix.hpp"
#include "geometry/pose_recovery.h"
#include "geometry/triangulation.hpp"

#include "slam/visual_odometry.hpp"

static std::vector<std::filesystem::path> loadImagePaths(const std::filesystem::path &folder)
{
    std::vector<std::filesystem::path> imagePaths;
    if (!std::filesystem::is_directory(folder))
    {
        return imagePaths;
    }

    for (const auto &entry : std::filesystem::directory_iterator(folder))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" || extension == ".tiff" || extension == ".tif")
        {
            imagePaths.push_back(entry.path());
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}

void testVisualOdometry()
{
    const std::filesystem::path sourceFolder = "/home/dk1242/Desktop/opencv/basics/VisionEngine/datasets/drive/image_00/data";
    auto imageFiles = loadImagePaths(sourceFolder);
    if (imageFiles.empty())
    {
        std::cerr << "No image files found in '" << sourceFolder << "'" << std::endl;
        return;
    }

    vision::FastCornerDetector fastDetector;
    vision::OrbExtractor orbExtractor;
    vision::DescriptorMatcher matcher;
    geometry::EssentialMatrixEstimator essentialEstimator;
    geometry::PoseRecovery poseRecovery;
    geometry::RansacFundamental fundamentalRansac;

    std::cout << "Loaded " << imageFiles.size() << " images from " << sourceFolder << std::endl;

    std::vector<cv::Point2d> previousPoints;
    std::vector<vision::OrbDescriptor> previousDescriptors;
    bool hasPreviousFrame = false;
    cv::Mat cameraMatrix;
    bool cameraInitialized = false;
    geometry::CameraPose recoveredPose;

    auto setPreviousFrame = [&](const std::vector<vision::KeyPoint> &keypoints,
                                const std::vector<vision::OrbDescriptor> &descriptors) {
        previousDescriptors = descriptors;
        previousPoints.clear();
        previousPoints.reserve(keypoints.size());
        for (const auto &keypoint : keypoints)
        {
            previousPoints.emplace_back(keypoint.position.x, keypoint.position.y);
        }
        hasPreviousFrame = true;
    };

    auto buildCameraMatrix = [](int width, int height) -> cv::Mat {
        const double focalLength = 800.0;
        const double cx = width * 0.5;
        const double cy = height * 0.5;
        cv::Mat K = (cv::Mat_<double>(3, 3) << focalLength, 0.0, cx,
                     0.0, focalLength, cy,
                     0.0, 0.0, 1.0);
        return K;
    };

    const size_t maxFrames = 20;
    const size_t frameCount = std::min(imageFiles.size(), maxFrames);

    for (size_t i = 0; i < frameCount; ++i)
    {
        const auto start = std::chrono::high_resolution_clock::now();
        const auto &imagePath = imageFiles[i];
        cv::Mat frame = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (frame.empty())
        {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }

        cv::Mat gray;
        if (frame.channels() == 3)
        {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }
        else if (frame.channels() == 4)
        {
            cv::cvtColor(frame, gray, cv::COLOR_BGRA2GRAY);
        }
        else
        {
            gray = frame.clone();
        }

        if (!cameraInitialized)
        {
            cameraMatrix = buildCameraMatrix(gray.cols, gray.rows);
            cameraInitialized = !cameraMatrix.empty();
        }

        std::vector<vision::KeyPoint> keypoints = fastDetector.detect(gray);
        keypoints = fastDetector.selectKeypointsInGrid(gray, keypoints, 8, 8, 300);
        orbExtractor.computeOrientations(gray, keypoints);
        std::vector<vision::OrbDescriptor> descriptors = orbExtractor.computeDescriptors(gray, keypoints);

        if (!hasPreviousFrame)
        {
            setPreviousFrame(keypoints, descriptors);
            std::cout << "Loaded first frame with " << keypoints.size() << " keypoints" << std::endl;
            continue;
        }

        auto knnForward = matcher.knnMatch(previousDescriptors, descriptors, 2);
        auto ratioMatches = matcher.ratioTest(knnForward, 0.75f);
        auto knnBackward = matcher.knnMatch(descriptors, previousDescriptors, 2);
        auto ratioBackwardMatches = matcher.ratioTest(knnBackward, 0.75f);
        auto crossMatches = matcher.crosscheckMatch(ratioMatches, ratioBackwardMatches);

        std::vector<cv::Point2d> points1;
        std::vector<cv::Point2d> points2;
        points1.reserve(crossMatches.size());
        points2.reserve(crossMatches.size());

        for (const auto &match : crossMatches)
        {
            points1.push_back(previousPoints[match.queryIdx]);
            points2.emplace_back(keypoints[match.trainIdx].position.x, keypoints[match.trainIdx].position.y);
        }

        if (crossMatches.size() < 30)
        {
            const auto end = std::chrono::high_resolution_clock::now();
            const double elapsedMs =
                std::chrono::duration<double, std::milli>(end - start).count();

            std::cout << "Frame " << i << " -> " << (i + 1)
                      << " | Matches: " << crossMatches.size()
                      << " | Skipped: insufficient matches"
                      << " | Time: " << std::fixed << std::setprecision(1)
                      << elapsedMs << " ms" << std::endl;
            setPreviousFrame(keypoints, descriptors);
            continue;
        }

        const auto ransacResult = fundamentalRansac.estimateFundamental(points1, points2);
        const cv::Mat &F = ransacResult.model;
        const cv::Mat &inlierMask = ransacResult.inlierMask;
        const size_t ransacInliers = ransacResult.inlierCount;

        std::vector<cv::Point2d> inlierPoints1;
        std::vector<cv::Point2d> inlierPoints2;
        inlierPoints1.reserve(ransacInliers);
        inlierPoints2.reserve(ransacInliers);
        for (int k = 0; k < inlierMask.rows; ++k)
        {
            if (inlierMask.at<uchar>(k))
            {
                inlierPoints1.push_back(points1[static_cast<size_t>(k)]);
                inlierPoints2.push_back(points2[static_cast<size_t>(k)]);
            }
        }

        bool validPose = ransacInliers >= 20;
        int positiveDepthCount = 0;
        double rotationAngleDeg = 0.0;
        double translationNorm = 0.0;
        cv::Vec3d translationDirection(0.0, 0.0, 0.0);
        std::string poseFailureReason;
        cv::Mat E;
        if (F.empty() || F.rows != 3 || F.cols != 3)
        {
            validPose = false;
            poseFailureReason = "invalid fundamental matrix";
        }
        else if (ransacInliers < 20)
        {
            validPose = false;
            poseFailureReason = "insufficient inliers";
        }
        else
        {
            if (cameraInitialized)
            {
                E = essentialEstimator.computeEssentialMatrix(F, cameraMatrix);
                E = essentialEstimator.enforceEssentialConstraints(E);
                if (E.empty())
                {
                    validPose = false;
                    poseFailureReason = "invalid essential matrix";
                }
                else
                {
                    recoveredPose = poseRecovery.recoverPose(
                        E, inlierPoints1, inlierPoints2, cameraMatrix);
                    positiveDepthCount = poseRecovery.countPositiveDepth(
                        recoveredPose, inlierPoints1, inlierPoints2, cameraMatrix);

                    cv::Mat rvec;
                    cv::Rodrigues(recoveredPose.R, rvec);
                    double rotationAngle = cv::norm(rvec);
                    rotationAngleDeg = rotationAngle * 180.0 / CV_PI;
                    translationNorm = cv::norm(recoveredPose.t);
                    const cv::Vec3d rawTranslation(
                        recoveredPose.t.at<double>(0),
                        recoveredPose.t.at<double>(1),
                        recoveredPose.t.at<double>(2));
                    if (translationNorm > std::numeric_limits<double>::epsilon())
                    {
                        translationDirection = rawTranslation / translationNorm;
                    }
                }
            }
            else
            {
                validPose = false;
                poseFailureReason = "invalid camera matrix";
            }
        }

        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsedMs =
            std::chrono::duration<double, std::milli>(end - start).count();

        std::cout << std::fixed << std::setprecision(2)
                  << "Frame " << i << " -> " << (i + 1)
                  << " | Matches: " << crossMatches.size()
                  << " | Inliers: " << ransacInliers
                  << " | Positive Depth: " << positiveDepthCount
                  << " | Rot: " << std::setprecision(2) << rotationAngleDeg << " deg"
                  << " | t: ["
                  << translationDirection[0] << ", "
                  << translationDirection[1] << ", "
                  << translationDirection[2] << "]"
                  << " | Time: " << std::setprecision(1) << elapsedMs << " ms";
        if (!validPose)
        {
            std::cout << " | Pose: invalid (" << poseFailureReason << ")";
        }
        std::cout << std::endl;

        setPreviousFrame(keypoints, descriptors);
    }
}

int main(int argc, char **argv)
{
    testVisualOdometry();
    return 0;
}
