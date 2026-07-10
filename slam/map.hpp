#pragma once

#include <opencv2/opencv.hpp>
#include <unordered_map>

#include "slam/feature_track.hpp"

namespace slam
{
    // A persistent 3D landmark. Created once by Mapping when a FeatureTrack
    // has enough evidence to triangulate reliably; subsequent observations
    // of the same track refine it rather than creating a new one.
    struct MapPoint
    {
        MapPointId id;
        cv::Point3d position;
        size_t observationCount = 1;
        bool active = true;
    };

    // Owns the set of MapPoints, keyed by MapPointId so identity never
    // depends on storage position -- mirrors TrackManager's tracks_.
    class Map
    {
    public:
        MapPointId createPoint(const cv::Point3d &position, size_t observationCount);

        const MapPoint *find(MapPointId id) const;
        void remove(MapPointId id);

        const std::unordered_map<MapPointId, MapPoint> &points() const;
        size_t size() const;

    private:
        std::unordered_map<MapPointId, MapPoint> points_;
        MapPointId nextId_ = 0;
    };
}
