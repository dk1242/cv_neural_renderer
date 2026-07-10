#pragma once

#include <opencv2/opencv.hpp>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "vision/orb.hpp"

namespace slam
{
    using TrackId = std::uint64_t;
    using FrameId = std::uint64_t;
    using MapPointId = std::uint64_t;

    struct Observation
    {
        FrameId frameId;
        size_t keypointIndex;
        cv::Point2f pixel;
        vision::OrbDescriptor descriptor;
    };

    // Camera pose in world frame (camera-to-world), as opposed to
    // geometry::CameraPose which stores a relative (world-to-camera-style)
    // R/t between two views. Shared between VisualOdometry and Mapping so
    // both can hand poses to one another without a header cycle.
    struct CameraPose
    {
        cv::Mat Rwc;
        cv::Mat twc;
    };

    // One inlier match expressed as the pair of Observations it produced,
    // one in the previous frame and one in the current frame. This is the
    // vocabulary TrackManager operates on -- callers translate
    // detector/matcher output (keypoints, descriptors, matches) into
    // Correspondences so TrackManager itself never needs to know about
    // vision:: types.
    struct Correspondence
    {
        Observation previous;
        Observation current;
    };

    class FeatureTrack
    {
    public:
        FeatureTrack(TrackId id, Observation firstObservation);

        TrackId id() const;
        size_t length() const;
        const std::vector<Observation> &observations() const;
        const Observation &lastObservation() const;

        void addObservation(Observation observation);

        const std::optional<MapPointId> &mapPointId() const;
        void setMapPointId(MapPointId id);

        bool active() const;
        void setActive(bool active);

    private:
        TrackId id_;
        std::vector<Observation> observations_;
        std::optional<MapPointId> mapPointId_;
        bool active_ = true;
    };
}
