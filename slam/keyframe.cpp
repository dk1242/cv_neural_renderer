#include "slam/keyframe.hpp"

#include <utility>

slam::Keyframe::Keyframe(KeyframeId id, FrameId sourceFrameId, CameraPose pose,
                          std::vector<vision::KeyPoint> keypoints,
                          std::vector<vision::OrbDescriptor> descriptors)
    : id_(id), sourceFrameId_(sourceFrameId), pose_(std::move(pose)),
      keypoints_(std::move(keypoints)), descriptors_(std::move(descriptors))
{
}

slam::KeyframeId slam::Keyframe::id() const
{
    return id_;
}

slam::FrameId slam::Keyframe::sourceFrameId() const
{
    return sourceFrameId_;
}

const slam::CameraPose &slam::Keyframe::pose() const
{
    return pose_;
}

const std::vector<vision::KeyPoint> &slam::Keyframe::keypoints() const
{
    return keypoints_;
}

const std::vector<vision::OrbDescriptor> &slam::Keyframe::descriptors() const
{
    return descriptors_;
}
