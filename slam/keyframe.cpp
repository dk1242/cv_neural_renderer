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

void slam::Keyframe::addObservation(MapPointId mapPointId, const MapObservation &observation)
{
    observations_[mapPointId] = observation;
}

void slam::Keyframe::removeObservation(MapPointId mapPointId)
{
    observations_.erase(mapPointId);
}

bool slam::Keyframe::hasObservation(MapPointId mapPointId) const
{
    return observations_.find(mapPointId) != observations_.end();
}

const slam::MapObservation *slam::Keyframe::observation(MapPointId mapPointId) const
{
    const auto it = observations_.find(mapPointId);
    return it == observations_.end() ? nullptr : &it->second;
}

size_t slam::Keyframe::numObservations() const
{
    return observations_.size();
}
