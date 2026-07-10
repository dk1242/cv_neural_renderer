#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <utility>
#include <vector>

#include "geometry/pose_recovery.h"
#include "geometry/triangulation.hpp"
#include "slam/map.hpp"
#include "slam/track_manager.hpp"

namespace slam
{
    // Grows the persistent world model from FeatureTracks. Promotes a track
    // into a MapPoint the first time it has a fresh two-view correspondence
    // with positive depth; tracks that already reference a MapPoint are
    // left alone, so landmarks are triangulated once, not every frame.
    class Mapping
    {
    public:
        struct UpdateStats
        {
            size_t newMapPoints = 0;
            // (trackId, mapPointId) for every track promoted this call.
            std::vector<std::pair<TrackId, MapPointId>> promotions;
        };

        UpdateStats update(TrackManager &trackManager, FrameId currentFrameId,
                           const geometry::CameraPose &relativePose,
                           const CameraPose &previousWorldPose,
                           const cv::Mat &cameraMatrix);

        const Map &map() const;

    private:
        std::optional<cv::Point3d> triangulateWorldPoint(
            const cv::Point2d &pixelPrev, const cv::Point2d &pixelCurr,
            const geometry::CameraPose &pose1, const geometry::CameraPose &pose2,
            const cv::Mat &cameraMatrix, const CameraPose &previousWorldPose);

        Map map_;
        geometry::Triangulator triangulator_;
    };
}
