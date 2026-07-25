#include "slam/localmapping.hpp"

#include <unordered_set>

#include "slam/map.hpp"

constexpr size_t kMinCovisibilityWeight = 5;

slam::LocalMapping::LocalMapping(Mapping &mapping) : mapping_(mapping)
{
}

void slam::LocalMapping::insertKeyframe(KeyframeId id)
{
    keyFrameQueue_.push(id);
}

void slam::LocalMapping::process()
{
    KeyframeId id = keyFrameQueue_.front();
    keyFrameQueue_.pop();

    processNewKeyframe(id);
}

void slam::LocalMapping::processNewKeyframe(KeyframeId id)
{
    const slam::Keyframe *keyframe = mapping_.map().getKeyframe(id);

    if (!keyframe)
        return;

    const auto localKeyframes = selectLocalKeyframes(id);
    const auto localMapPoints = selectLocalMapPoints(localKeyframes);

    std::cout << "Processing KF " << id
              << " -- local KFs: " << localKeyframes.all.size()
              << ", local MapPoints: " << localMapPoints.all.size() << '\n';
}

slam::LocalKeyFrameSet slam::LocalMapping::selectLocalKeyframes(KeyframeId current)
{
    LocalKeyFrameSet result;
    result.current = current;

    std::unordered_set<KeyframeId> visited{current};
    const auto& graph = mapping_.covisibilityGraph();

    for (const auto &neighbor : graph.neighbors(current))
    {
        if (neighbor.weight < kMinCovisibilityWeight)
        {
            continue;
        }
        if (visited.insert(neighbor.keyframeId).second)
        {
            result.firstOrder.push_back(neighbor.keyframeId);
        }
    }

    for (const auto &firstOrderId : result.firstOrder)
    {
        for (const auto &neighbor : graph.neighbors(firstOrderId))
        {
            if (visited.insert(neighbor.keyframeId).second)
            {
                result.secondOrder.push_back(neighbor.keyframeId);
            }
        }
    }

    result.all.push_back(current);
    result.all.insert(result.all.end(), result.firstOrder.begin(), result.firstOrder.end());
    result.all.insert(result.all.end(), result.secondOrder.begin(), result.secondOrder.end());

    return result;
}

slam::LocalMapPointSet slam::LocalMapping::selectLocalMapPoints(const LocalKeyFrameSet &localKeyframes)
{
    LocalMapPointSet result;
    std::unordered_set<MapPointId> visited;

    for (const auto &keyframeId : localKeyframes.all)
    {
        const slam::Keyframe *keyframe = mapping_.map().getKeyframe(keyframeId);
        if (!keyframe)
        {
            continue;
        }

        for (const auto &[mapPointId, observation] : keyframe->observations())
        {
            if (visited.insert(mapPointId).second)
            {
                result.all.push_back(mapPointId);
            }
        }
    }

    return result;
}