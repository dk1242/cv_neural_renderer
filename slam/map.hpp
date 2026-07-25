#pragma once

#include <opencv2/opencv.hpp>
#include <optional>
#include <unordered_map>

#include "slam/feature_track.hpp"
#include "slam/keyframe.hpp"
#include "slam/map_observation.hpp"

namespace slam
{
    // A persistent 3D landmark. Created once by Mapping when a FeatureTrack
    // has enough evidence to triangulate reliably; subsequent observations
    // of the same track refine it rather than creating a new one.
    class MapPoint
    {
    public:
        MapPoint(MapPointId id, cv::Point3d position, size_t observationCount = 1, bool active = true);

        MapPointId id;
        cv::Point3d position;
        size_t observationCount = 1;
        bool active = true;

        // Per-Keyframe sightings of this point, keyed by KeyframeId. Kept
        // private -- mutate only through these methods -- so a caller can't
        // desync the map from the KeyframeId it's keyed under. Returns
        // whether this KeyframeId was new (false if it already had an
        // observation here, which is just refreshed in place).
        bool addObservation(const MapObservation& observation);
        void removeObservation(KeyframeId keyframeId);
        bool hasObservation(KeyframeId keyframeId) const;
        const MapObservation *observation(KeyframeId keyframeId) const;
        size_t numObservations() const;
        const std::unordered_map<KeyframeId, MapObservation> &observations() const;

    private:
        std::unordered_map<KeyframeId, MapObservation> observations_;
    };

    // Owns the set of MapPoints, keyed by MapPointId so identity never
    // depends on storage position -- mirrors TrackManager's tracks_.
    class Map
    {
    public:
        MapPointId createPoint(const cv::Point3d &position, size_t observationCount);

        const MapPoint *find(MapPointId id) const;
        void remove(MapPointId id);

        // Adds an observation to an already-created MapPoint. No-op (returns
        // false) if mapPointId is unknown -- keeps the "invariants live
        // inside MapPoint" rule intact while letting Mapping stay ignorant of
        // MapPoint's internals. Otherwise returns whatever MapPoint::addObservation
        // reports, i.e. whether this was a genuinely new sighting.
        bool addObservation(MapPointId mapPointId, const MapObservation &observation);

        const std::unordered_map<MapPointId, MapPoint> &points() const;
        size_t size() const;

        KeyframeId createKeyframe(FrameId sourceFrameId, CameraPose pose,
                                  std::vector<vision::KeyPoint> keypoints,
                                  std::vector<vision::OrbDescriptor> descriptors);
        const Keyframe *getKeyframe(KeyframeId id) const;
        void removeKeyframe(KeyframeId id);

        const std::unordered_map<KeyframeId, Keyframe> &keyframes() const;

        // Bridge from a source Frame back to the KeyFrame it became, if any.
        // Most Frames are tracking-only and never appear here -- only Frames
        // that passed shouldCreateKeyframe() have an entry.
        std::optional<KeyframeId> keyframeForFrame(FrameId frameId) const;

    private:
        std::unordered_map<MapPointId, MapPoint> points_;
        MapPointId nextId_ = 0;

        std::unordered_map<KeyframeId, Keyframe> keyframes_;
        KeyframeId nextKeyframeId_ = 0;

        std::unordered_map<FrameId, KeyframeId> frameToKeyframe_;
    };
}
