#pragma once

#include <unordered_map>
#include <vector>

#include "slam/feature_track.hpp"
#include "slam/map_observation.hpp"
#include "vision/fast.hpp"
#include "vision/orb.hpp"

namespace slam
{

// A persistent snapshot of a Frame: pose, keypoints, and descriptors as they
// were at the moment the frame was promoted to a KeyFrame. Nothing else --
// no observations, no MapPoint links, no covisibility -- those come once
// KeyFrames are wired into the observation graph.
class Keyframe
{
public:
    Keyframe(KeyframeId id, FrameId sourceFrameId, CameraPose pose,
             std::vector<vision::KeyPoint> keypoints,
             std::vector<vision::OrbDescriptor> descriptors);

    KeyframeId id() const;
    FrameId sourceFrameId() const;
    const CameraPose &pose() const;
    const std::vector<vision::KeyPoint> &keypoints() const;
    const std::vector<vision::OrbDescriptor> &descriptors() const;

    // Per-MapPoint sightings recorded in this KeyFrame, keyed by MapPointId
    // -- the mirror image of MapPoint's own KeyframeId-keyed observation
    // map. Kept private for the same reason: mutate only through these
    // methods so the map can't desync from the id it's keyed under.
    void addObservation(MapPointId mapPointId, const MapObservation &observation);
    void removeObservation(MapPointId mapPointId);
    bool hasObservation(MapPointId mapPointId) const;
    const MapObservation *observation(MapPointId mapPointId) const;
    size_t numObservations() const;

private:
    KeyframeId id_;
    FrameId sourceFrameId_;
    CameraPose pose_;
    std::vector<vision::KeyPoint> keypoints_;
    std::vector<vision::OrbDescriptor> descriptors_;
    std::unordered_map<MapPointId, MapObservation> observations_;
};

}
