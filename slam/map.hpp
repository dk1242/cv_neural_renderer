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

        // Average unit viewing direction (observing camera center ->
        // position) across all current observations -- kept up to date by
        // Map::addObservation()/removeObservation(). Zero vector means "not
        // yet established" (e.g. a single observation); callers should skip
        // any check that depends on it rather than treat it as a real
        // direction.
        cv::Point3d normal = cv::Point3d(0.0, 0.0, 0.0);

        // Range of observing-camera-to-point distances this MapPoint has
        // actually been seen at, across all current observations -- kept up
        // to date alongside normal. A candidate view whose distance falls
        // far outside this range is seeing the point at a scale it's never
        // been matched at before, so its descriptor is unlikely to match
        // reliably. ORB-SLAM derives this from the ORB pyramid octave of
        // each observation instead; we don't track per-point octave yet, so
        // this is the plain observed min/max as a stand-in. minDistance ==
        // maxDistance == 0.0 means "not yet established" (e.g. a single
        // observation) -- skip the check rather than treat it as a real
        // range.
        double minDistance = 0.0;
        double maxDistance = 0.0;

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

        // Symmetric counterpart to addObservation() -- drops the sighting
        // from both the MapPoint and the KeyFrame side so neither is left
        // referencing the other. No-op if either id is unknown.
        void removeObservation(MapPointId mapPointId, KeyframeId keyframeId);

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
        // Recomputes a MapPoint's normal and min/maxDistance together, in
        // one pass over its current observations' KeyFrame camera centers
        // -- mirrors ORB-SLAM's combined UpdateNormalAndDepth(). Called
        // internally by addObservation()/removeObservation() so neither
        // field ever goes stale; there is no public way to trigger this
        // separately.
        void updateNormalAndDistance(MapPointId mapPointId);

        std::unordered_map<MapPointId, MapPoint> points_;
        MapPointId nextId_ = 0;

        std::unordered_map<KeyframeId, Keyframe> keyframes_;
        KeyframeId nextKeyframeId_ = 0;

        std::unordered_map<FrameId, KeyframeId> frameToKeyframe_;
    };
}
