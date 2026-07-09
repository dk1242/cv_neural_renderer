#include "feature_track.hpp"

#include <utility>

slam::FeatureTrack::FeatureTrack(TrackId id, Observation firstObservation)
    : id_(id)
{
    observations_.push_back(std::move(firstObservation));
}

slam::TrackId slam::FeatureTrack::id() const
{
    return id_;
}

size_t slam::FeatureTrack::length() const
{
    return observations_.size();
}

const std::vector<slam::Observation> &slam::FeatureTrack::observations() const
{
    return observations_;
}

const slam::Observation &slam::FeatureTrack::lastObservation() const
{
    return observations_.back();
}

void slam::FeatureTrack::addObservation(Observation observation)
{
    observations_.push_back(std::move(observation));
}

const std::optional<slam::MapPointId> &slam::FeatureTrack::mapPointId() const
{
    return mapPointId_;
}

void slam::FeatureTrack::setMapPointId(MapPointId id)
{
    mapPointId_ = id;
}

bool slam::FeatureTrack::active() const
{
    return active_;
}

void slam::FeatureTrack::setActive(bool active)
{
    active_ = active;
}
