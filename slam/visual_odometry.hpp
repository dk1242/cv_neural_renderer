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
#include "slam/map.hpp"
#include "slam/mapping.hpp"
#include "slam/track_manager.hpp"

namespace slam
{
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
        const Map& map() const;
        const std::vector<cv::Point3d>& cameraCenters() const;
        const CameraPose& currentPose() const;
        const TrackManager& trackManager() const;

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

        vision::FastCornerDetector fastDetector_;
        vision::OrbExtractor orbExtractor_;
        vision::DescriptorMatcher matcher_;
        geometry::RansacFundamental fundamentalRansac_;
        geometry::EssentialMatrixEstimator essentialEstimator_;
        geometry::PoseRecovery poseRecovery_;
        TrackManager trackManager_;
        Mapping mapping_;

        Frame createFrame(const cv::Mat &image);
        void setPreviousFrame(Frame frame);
        Observation makeObservation(FrameId frameId, size_t keypointIndex, const Frame &frame) const;
        void logFrameStats(size_t matchCount, std::optional<size_t> inlierCount,
                            const TrackManager::UpdateStats &trackStats,
                            const std::string &statusLine, double elapsedMs) const;
        void logMappingStats(const Mapping::UpdateStats &mappingStats) const;
    };
}
