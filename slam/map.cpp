#include "slam/map.hpp"

slam::MapPoint::MapPoint(MapPointId id, cv::Point3d position, size_t observationCount, bool active)
    : id(id), position(position), observationCount(observationCount), active(active)
{
}

void slam::MapPoint::addObservation(const MapObservation& observation)
{
    observations_[observation.keyframeId] = observation;
}

void slam::MapPoint::removeObservation(KeyframeId keyframeId)
{
    observations_.erase(keyframeId);
}

bool slam::MapPoint::hasObservation(KeyframeId keyframeId) const
{
    return observations_.find(keyframeId) != observations_.end();
}

const slam::MapObservation *slam::MapPoint::observation(KeyframeId keyframeId) const
{
    const auto it = observations_.find(keyframeId);
    return it == observations_.end() ? nullptr : &it->second;
}

size_t slam::MapPoint::numObservations() const
{
    return observations_.size();
}

const std::unordered_map<slam::KeyframeId, slam::MapObservation> &slam::MapPoint::observations() const
{
    return observations_;
}

slam::MapPointId slam::Map::createPoint(const cv::Point3d &position, size_t observationCount)
{
    const MapPointId id = nextId_++;
    points_.emplace(id, MapPoint{id, position,
                                 observationCount, true});
    return id;
}

const slam::MapPoint *slam::Map::find(MapPointId id) const
{
    const auto it = points_.find(id);
    return it == points_.end() ? nullptr : &it->second;
}

void slam::Map::remove(MapPointId id)
{
    points_.erase(id);
}

void slam::Map::addObservation(MapPointId mapPointId, const MapObservation &observation)
{
    const auto pointIt = points_.find(mapPointId);
    if (pointIt == points_.end())
    {
        return;
    }

    pointIt->second.addObservation(observation);

    const auto keyframeIt = keyframes_.find(observation.keyframeId);
    if (keyframeIt != keyframes_.end())
    {
        keyframeIt->second.addObservation(mapPointId, observation);
    }
}

const std::unordered_map<slam::MapPointId, slam::MapPoint> &slam::Map::points() const
{
    return points_;
}

size_t slam::Map::size() const
{
    return points_.size();
}

slam::KeyframeId slam::Map::createKeyframe(FrameId sourceFrameId, CameraPose pose,
                                            std::vector<vision::KeyPoint> keypoints,
                                            std::vector<vision::OrbDescriptor> descriptors)
{
    const KeyframeId id = nextKeyframeId_++;
    keyframes_.emplace(id, Keyframe{id, sourceFrameId, std::move(pose),
                                     std::move(keypoints), std::move(descriptors)});
    frameToKeyframe_[sourceFrameId] = id;
    return id;
}

const slam::Keyframe *slam::Map::getKeyframe(KeyframeId id) const
{
    const auto it = keyframes_.find(id);
    return it == keyframes_.end() ? nullptr : &it->second;
}

void slam::Map::removeKeyframe(KeyframeId id)
{
    keyframes_.erase(id);
}

const std::unordered_map<slam::KeyframeId, slam::Keyframe> &slam::Map::keyframes() const
{
    return keyframes_;
}

std::optional<slam::KeyframeId> slam::Map::keyframeForFrame(FrameId frameId) const
{
    const auto it = frameToKeyframe_.find(frameId);
    return it == frameToKeyframe_.end() ? std::nullopt : std::optional<KeyframeId>(it->second);
}
