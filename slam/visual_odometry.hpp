#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <vector>

#include "geometry/essential_matrix.hpp"
#include "vision/fast.hpp"
#include "vision/orb.hpp"
#include "vision/descriptor_matcher.h"
#include "geometry/pose_recovery.h"
#include "geometry/ransac.hpp"
#include "geometry/triangulation.hpp"

namespace slam
{
    struct MapPoint
    {
        cv::Point3d position;
    };
    struct CameraPose
    {
        cv::Mat Rwc;
        cv::Mat twc;
    };
    struct Frame
    {
        cv::Mat image;
        std::vector<vision::KeyPoint> keypoints;
        std::vector<vision::OrbDescriptor> descriptors;
    };
    struct RelativePose
    {
        cv::Mat R;
        cv::Mat t;
        size_t inliers = 0;
    };

    class VisualOdometry
    {
    public:
        void processFrame(const cv::Mat &frame);
        const std::optional<RelativePose> &lastRelativePose() const;
        void setCameraCenters();
        cv::Mat drawTrajectory() const;
        const std::vector<MapPoint>& mapPoints() const;
        const std::vector<cv::Point3d>& cameraCenters() const;
        const CameraPose& currentPose() const;

    private:
        Frame previousFrame_;
        bool initialized_ = false;
        cv::Mat cameraMatrix_;
        size_t frameIndex_ = 0;
        std::optional<RelativePose> lastRelativePose_;
        slam::CameraPose currentPose_;
        slam::CameraPose previousPose_;
        std::vector<CameraPose> trajectory_;
        std::vector<cv::Point3d> cameraCenters_;
        std::vector<MapPoint> mapPoints_;

        vision::FastCornerDetector fastDetector_;
        vision::OrbExtractor orbExtractor_;
        vision::DescriptorMatcher matcher_;
        geometry::RansacFundamental fundamentalRansac_;
        geometry::EssentialMatrixEstimator essentialEstimator_;
        geometry::PoseRecovery poseRecovery_;
        geometry::Triangulator triangulator_;

        Frame createFrame(const cv::Mat &image);
        void setPreviousFrame(Frame frame);
        void triangulateCurrentFrame(const geometry::CameraPose &relativePose, std::vector<cv::Point2d> &inlierPoints1, std::vector<cv::Point2d> &inlierPoints2);
    };
}
