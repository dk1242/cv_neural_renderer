#pragma once

#include <vector>

#include "slam/feature_track.hpp"
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

private:
    KeyframeId id_;
    FrameId sourceFrameId_;
    CameraPose pose_;
    std::vector<vision::KeyPoint> keypoints_;
    std::vector<vision::OrbDescriptor> descriptors_;
};

}
