#include "slam/map.hpp"

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
