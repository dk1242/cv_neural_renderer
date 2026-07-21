#pragma once

#include <opencv2/opencv.hpp>
#include <cstddef>

#include "slam/feature_track.hpp"

namespace slam
{
    // A single sighting of a MapPoint in one Keyframe: which keypoint slot
    // in that Keyframe's feature list produced it, and where it landed in
    // pixel space. Keyed by KeyframeId inside MapPoint's observation map.
    struct MapObservation
    {
        KeyframeId keyframeId;
        size_t keypointIndex;
        cv::Point2f pixel;
    };
}
