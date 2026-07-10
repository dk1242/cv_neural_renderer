#pragma once

#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "slam/feature_track.hpp"

namespace slam
{
    // Maintains the set of FeatureTracks, keyed by TrackId so identity never
    // depends on storage position. Operates purely on slam:: types
    // (Observation/Correspondence) -- callers are responsible for turning
    // detector/matcher output into Correspondences.
    class TrackManager
    {
    public:
        // What happened to tracks during one update() call, as opposed to
        // activeTrackCount()/tracks() which describe cumulative state.
        struct UpdateStats
        {
            size_t newTracks = 0;
            size_t extendedTracks = 0;
            size_t terminatedTracks = 0;
        };

        struct LengthStats
        {
            size_t longest = 0;
            double average = 0.0;
        };

        UpdateStats update(const std::vector<Correspondence> &correspondences);

        // Links an existing track to a MapPoint once Mapping has promoted
        // it. Kept as a setter on TrackManager (rather than exposing
        // mutable tracks()) so FeatureTrack ownership stays exclusively
        // with TrackManager.
        bool linkTrackToMapPoint(TrackId trackId, MapPointId mapPointId);

        const std::unordered_map<TrackId, FeatureTrack> &tracks() const;
        size_t activeTrackCount() const;
        size_t terminatedTrackCount() const;
        size_t totalTrackCount() const;
        LengthStats activeTrackLengthStats() const;

    private:
        // Looks up the track currently ending at this previous-frame
        // keypoint index, if any.
        std::optional<TrackId> findTrack(size_t previousKeypointIndex) const;
        void extendTrack(TrackId trackId, Observation observation);
        TrackId createTrack(Observation firstObservation, Observation secondObservation);
        // Any track live going into this update() but not present in
        // extendedTrackIds wasn't matched again this frame -- flip it
        // inactive in place. Returns how many tracks were terminated.
        size_t terminateStaleTracks(const std::unordered_set<TrackId> &extendedTrackIds);

        std::unordered_map<TrackId, FeatureTrack> tracks_;
        // Maps a keypoint index in the previous frame to the TrackId it
        // currently terminates, so the next update() can look up "does this
        // match continue a track" without depending on any storage index.
        // Rebuilt every call.
        std::unordered_map<size_t, TrackId> previousKeypointToTrack_;
        TrackId nextTrackId_ = 0;
    };
}
