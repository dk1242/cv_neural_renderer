#pragma once

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "slam/feature_track.hpp"

namespace slam
{
    class Map;

    struct CovisibilityEdge
    {
        KeyframeId keyframeId;
        size_t weight = 0;
    };

    // Tracks how many MapPoints each pair of Keyframes observes in common.
    // Rebuilt from scratch on demand from the current Map state -- it does
    // not incrementally track changes as Keyframes/MapPoints are added or
    // removed.
    class CovisibilityGraph
    {
    public:
        void rebuild(const Map &map);

        // Sorted strongest-first (descending weight) -- Local Mapping wants
        // the best-covisible neighbors, not an arbitrary order.
        std::vector<CovisibilityEdge> neighbors(KeyframeId keyframeId) const;

        size_t weight(KeyframeId a, KeyframeId b) const;

        bool contains(KeyframeId id) const;

    private:
        std::unordered_map<KeyframeId, std::unordered_map<KeyframeId, size_t>> adjacency_;
    };
}
