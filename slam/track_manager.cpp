#include "track_manager.hpp"

#include <algorithm>
#include <utility>

std::optional<slam::TrackId> slam::TrackManager::findTrack(size_t previousKeypointIndex) const
{
    const auto it = previousKeypointToTrack_.find(previousKeypointIndex);
    if (it == previousKeypointToTrack_.end())
    {
        return std::nullopt;
    }
    return it->second;
}

void slam::TrackManager::extendTrack(TrackId trackId, Observation observation)
{
    tracks_.at(trackId).addObservation(std::move(observation));
}

slam::TrackId slam::TrackManager::createTrack(Observation firstObservation, Observation secondObservation)
{
    const TrackId newId = nextTrackId_++;
    FeatureTrack track(newId, std::move(firstObservation));
    track.addObservation(std::move(secondObservation));
    tracks_.emplace(newId, std::move(track));
    return newId;
}

size_t slam::TrackManager::terminateStaleTracks(const std::unordered_set<TrackId> &extendedTrackIds)
{
    // Checked against track state directly, not derived from
    // previousKeypointToTrack_'s contents -- staleness is a property of a
    // track, not an artifact of how the lookup map happens to be built.
    size_t terminatedCount = 0;
    for (auto &[id, track] : tracks_)
    {
        if (!track.active())
        {
            continue;
        }

        if (extendedTrackIds.count(id))
        {
            continue;
        }

        track.setActive(false);
        ++terminatedCount;
    }
    return terminatedCount;
}

slam::TrackManager::UpdateStats slam::TrackManager::update(const std::vector<Correspondence> &correspondences)
{
    std::unordered_map<size_t, TrackId> newPreviousKeypointToTrack;
    std::unordered_set<TrackId> extendedTrackIds;
    // Guards against a degenerate matcher output where the same previous
    // keypoint, or the same current keypoint, appears in more than one
    // correspondence -- without this, two correspondences sharing a
    // current keypoint would both add an observation to a real track, but
    // only the second write into newPreviousKeypointToTrack would survive,
    // silently orphaning the first track (still active, but unreachable by
    // any future lookup).
    std::unordered_set<size_t> usedPreviousKeypoints;
    std::unordered_set<size_t> usedCurrentKeypoints;

    UpdateStats stats;

    for (const auto &correspondence : correspondences)
    {
        const size_t previousKeypointIndex = correspondence.previous.keypointIndex;
        const size_t currentKeypointIndex = correspondence.current.keypointIndex;

        if (!usedPreviousKeypoints.insert(previousKeypointIndex).second)
        {
            continue;
        }

        if (!usedCurrentKeypoints.insert(currentKeypointIndex).second)
        {
            continue;
        }

        TrackId trackId;
        if (const auto existingTrack = findTrack(previousKeypointIndex))
        {
            trackId = *existingTrack;
            extendTrack(trackId, correspondence.current);
            ++stats.extendedTracks;
        }
        else
        {
            trackId = createTrack(correspondence.previous, correspondence.current);
            ++stats.newTracks;
        }

        extendedTrackIds.insert(trackId);
        newPreviousKeypointToTrack[currentKeypointIndex] = trackId;
    }

    stats.terminatedTracks = terminateStaleTracks(extendedTrackIds);
    previousKeypointToTrack_ = std::move(newPreviousKeypointToTrack);

    return stats;
}

const std::unordered_map<slam::TrackId, slam::FeatureTrack> &slam::TrackManager::tracks() const
{
    return tracks_;
}

bool slam::TrackManager::linkTrackToMapPoint(TrackId trackId, MapPointId mapPointId)
{
    const auto it = tracks_.find(trackId);
    if (it == tracks_.end())
    {
        return false;
    }
    it->second.setMapPointId(mapPointId);
    return true;
}

size_t slam::TrackManager::activeTrackCount() const
{
    size_t count = 0;
    for (const auto &[id, track] : tracks_)
    {
        if (track.active())
        {
            ++count;
        }
    }
    return count;
}

size_t slam::TrackManager::terminatedTrackCount() const
{
    return tracks_.size() - activeTrackCount();
}

size_t slam::TrackManager::totalTrackCount() const
{
    return tracks_.size();
}

slam::TrackManager::LengthStats slam::TrackManager::activeTrackLengthStats() const
{
    size_t totalLength = 0;
    size_t activeCount = 0;
    LengthStats stats;

    for (const auto &[id, track] : tracks_)
    {
        if (!track.active())
        {
            continue;
        }

        const size_t length = track.length();
        totalLength += length;
        stats.longest = std::max(stats.longest, length);
        ++activeCount;
    }

    stats.average = activeCount > 0
                         ? static_cast<double>(totalLength) / static_cast<double>(activeCount)
                         : 0.0;
    return stats;
}
