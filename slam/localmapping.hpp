#pragma once

#include <queue>

#include "slam/feature_track.hpp"
#include "slam/mapping.hpp"
#include "slam/keyframe.hpp"

namespace slam
{

    // The ORB-SLAM "local window" around a KeyFrame. firstOrder is its
    // strongest covisibility neighbors (weight >= kMinCovisibilityWeight);
    // secondOrder expands one more hop from those without reapplying the
    // threshold. Extend with reference/fixed KeyFrame sets when local BA
    // needs to distinguish them -- current/firstOrder/secondOrder/all stay
    // source-compatible for existing consumers.
    struct LocalKeyFrameSet
    {
        KeyframeId current;

        std::vector<KeyframeId> firstOrder;
        std::vector<KeyframeId> secondOrder;
        std::vector<KeyframeId> all;
    };

    struct LocalMapPointSet
    {
        std::vector<MapPointId> all;
    };

    // Consumes KeyFrames handed off by VisualOdometry/Mapping off the main
    // tracking thread's critical path. Currently just queues them -- culling,
    // covisibility-driven point creation, and local BA are not implemented
    // yet (see process()).
    class LocalMapping
    {
    public:
        explicit LocalMapping(Mapping &mapping);

        void insertKeyframe(KeyframeId id);

        // TODO: drains keyFrameQueue_ and, for each KeyFrame, run map point
        // culling, new point creation against covisibility neighbors, and
        // local bundle adjustment. Currently just empties the queue.
        void process();

        void processNewKeyframe(KeyframeId id);

        // New point triangulation, MapPoint fusion, local BA, and KeyFrame
        // culling all operate on this set (and selectLocalMapPoints() below)
        // rather than querying the CovisibilityGraph independently.
        LocalKeyFrameSet selectLocalKeyframes(KeyframeId current);

        // Every MapPoint observed by any KeyFrame in localKeyframes.all.
        LocalMapPointSet selectLocalMapPoints(const LocalKeyFrameSet &localKeyframes);

    private:
        Mapping &mapping_;

        std::queue<KeyframeId> keyFrameQueue_;
    };
}
