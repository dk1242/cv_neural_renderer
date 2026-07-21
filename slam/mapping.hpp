#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <utility>
#include <vector>

#include "geometry/pose_recovery.h"
#include "geometry/triangulation.hpp"
#include "slam/covisibility_graph.hpp"
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

        // Motion-based keyframe insertion: true once the camera has moved
        // far enough (translation) or turned enough (rotation) since the
        // last KeyFrame that the new viewpoint is likely to add real
        // triangulation/mapping information. Always true before the first
        // KeyFrame exists. Will grow map-aware terms (tracked/new MapPoint
        // counts) once KeyFrames are linked to MapPoints via observations.
        bool shouldCreateKeyframe(const CameraPose &currentPose) const;

        KeyframeId createKeyframe(FrameId sourceFrameId, const CameraPose &pose,
                                  std::vector<vision::KeyPoint> keypoints,
                                  std::vector<vision::OrbDescriptor> descriptors);

        const Map &map() const;
        const CovisibilityGraph &covisibilityGraph() const;

    private:
        std::optional<cv::Point3d> triangulateWorldPoint(
            const cv::Point2d &pixelPrev, const cv::Point2d &pixelCurr,
            const geometry::CameraPose &pose1, const geometry::CameraPose &pose2,
            const cv::Mat &cameraMatrix, const CameraPose &previousWorldPose);

        Map map_;
        CovisibilityGraph covisibilityGraph_;
        geometry::Triangulator triangulator_;

        std::optional<CameraPose> lastKeyframePose_;
        double keyframeTranslationThreshold_ = 3.0;
        double keyframeRotationThresholdDeg_ = 15.0;
    };
}
