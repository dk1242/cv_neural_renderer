#include "slam/covisibility_graph.hpp"

#include <algorithm>

#include "slam/map.hpp"

void slam::CovisibilityGraph::rebuild(const Map &map)
{
    adjacency_.clear();

    for (const auto &[mapPointId, point] : map.points())
    {
        std::vector<KeyframeId> observers;
        observers.reserve(point.numObservations());
        for (const auto &[keyframeId, observation] : point.observations())
        {
            observers.push_back(keyframeId);
        }

        for (size_t i = 0; i < observers.size(); ++i)
        {
            for (size_t j = i + 1; j < observers.size(); ++j)
            {
                adjacency_[observers[i]][observers[j]]++;
                adjacency_[observers[j]][observers[i]]++;
            }
        }
    }
}

std::vector<slam::CovisibilityEdge> slam::CovisibilityGraph::neighbors(KeyframeId keyframeId) const
{
    std::vector<CovisibilityEdge> result;

    const auto it = adjacency_.find(keyframeId);
    if (it == adjacency_.end())
    {
        return result;
    }

    result.reserve(it->second.size());
    for (const auto &[neighborId, weight] : it->second)
    {
        result.push_back(CovisibilityEdge{neighborId, weight});
    }

    std::sort(result.begin(), result.end(),
              [](const CovisibilityEdge &a, const CovisibilityEdge &b)
              {
                  return a.weight > b.weight;
              });

    return result;
}

size_t slam::CovisibilityGraph::weight(KeyframeId a, KeyframeId b) const
{
    const auto it = adjacency_.find(a);
    if (it == adjacency_.end())
    {
        return 0;
    }

    const auto neighborIt = it->second.find(b);
    return neighborIt == it->second.end() ? 0 : neighborIt->second;
}

bool slam::CovisibilityGraph::contains(KeyframeId id) const
{
    return adjacency_.contains(id);
}
