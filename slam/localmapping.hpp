#pragma once

#include <optional>
#include <queue>

#include "geometry/epipolar_geometry.hpp"
#include "geometry/triangulation.hpp"
#include "slam/feature_track.hpp"
#include "slam/mapping.hpp"
#include "slam/keyframe.hpp"
#include "vision/descriptor_matcher.h"

namespace slam
{

    // The ORB-SLAM "local window" around a KeyFrame. firstOrder is its
    // strongest covisibility neighbors (weight >= kMinCovisibilityWeight);
    // secondOrder expands one more hop from those without reapplying the
    // threshold. Extend with reference/fixed KeyFrame sets when local BA
    // needs to distinguish them -- current/firstOrder/secondOrder/all stay
    // source-compatible for existing consumers.
    struct LocalKeyFrameSet
    {
        KeyframeId current;

        std::vector<KeyframeId> firstOrder;
        std::vector<KeyframeId> secondOrder;
        std::vector<KeyframeId> all;
    };

    struct LocalMapPointSet
    {
        std::vector<MapPointId> all;
    };

    struct TriangulationStats
    {
        size_t candidatePairs = 0;
        size_t candidateMatches = 0;

        size_t rejectedBaseline = 0;
        size_t rejectedParallax = 0;
        size_t rejectedCheirality = 0;
        size_t rejectedReprojection = 0;

        size_t insertedMapPoints = 0;
    };

    // Per-pair geometric state shared by every candidate correspondence
    // between `current` and `neighbor` -- computed once (buildPairContext())
    // and threaded through matching/triangulation instead of being
    // rederived per candidate. baseline/medianDepth are filled in by the
    // caller since they gate whether the (more expensive) rest of the
    // context is worth computing at all.
    struct KeyframePairContext
    {
        const Keyframe &current;
        const Keyframe &neighbor;

        geometry::CameraPose relativePose;
        cv::Mat fundamentalMatrix;

        geometry::CameraPose poseWorldToCamCurrent;
        geometry::CameraPose poseWorldToCamNeighbor;

        double baseline = 0.0;
        double medianDepth = 0.0;
    };

    // A candidate new-MapPoint match between the current KeyFrame and one
    // covisibility neighbor, prior to triangulation/validation.
    struct CandidateCorrespondence
    {
        KeyframeId currentKeyframeId;
        KeyframeId neighborKeyframeId;
        size_t currentKeypointIndex;
        size_t neighborKeypointIndex;
        cv::Point2d currentPixel;
        cv::Point2d neighborPixel;
    };

    enum class TriangulationRejection
    {
        kParallax,
        kCheirality,
        kReprojection,
    };

    // Result of triangulateCorrespondence(): point is set on success;
    // rejection explains why on failure (only meaningful when !point).
    struct TriangulationOutcome
    {
        std::optional<cv::Point3d> point;
        TriangulationRejection rejection;
    };

    // Consumes KeyFrames handed off by VisualOdometry/Mapping off the main
    // tracking thread's critical path. Currently just queues them -- culling,
    // covisibility-driven point creation, and local BA are not implemented
    // yet (see process()).
    class LocalMapping
    {
    public:
        explicit LocalMapping(Mapping &mapping);

        void insertKeyframe(KeyframeId id);

        // TODO: drains keyFrameQueue_ and, for each KeyFrame, run map point
        // culling, new point creation against covisibility neighbors, and
        // local bundle adjustment. Currently just empties the queue.
        void process();

        void processNewKeyframe(KeyframeId id);

        // New point triangulation, MapPoint fusion, local BA, and KeyFrame
        // culling all operate on this set (and selectLocalMapPoints() below)
        // rather than querying the CovisibilityGraph independently.
        LocalKeyFrameSet selectLocalKeyframes(KeyframeId current);

        // Every MapPoint observed by any KeyFrame in localKeyframes.all.
        LocalMapPointSet selectLocalMapPoints(const LocalKeyFrameSet &localKeyframes);

        // Triangulates new MapPoints between the current KeyFrame and its
        // first-order covisibility neighbors -- the ORB-SLAM
        // "CreateNewMapPoints" step.
        TriangulationStats triangulateNewMapPoints(KeyframeId current, const LocalKeyFrameSet &localKeyframes);

        // Projects every MapPoint in localMapPoints into currentKeyframeId
        // and either (a) records a missed observation, when the projected
        // feature slot isn't already claimed by a MapPoint, or (b) merges
        // the two MapPoints, when it's claimed by a *different* one -- the
        // ORB-SLAM "SearchInNeighbors"/Fuse step. Catches landmarks that got
        // independently triangulated more than once for the same physical
        // point.
        void fuseMapPoints(KeyframeId currentKeyframeId, const LocalMapPointSet &localMapPoints);

    private:
        geometry::CameraPose computeRelativePose(const Keyframe &from, const Keyframe &to) const;
        geometry::CameraPose worldToCameraPose(const CameraPose &pose) const;
        double computeMedianSceneDepth(const Keyframe &keyframe) const;

        // Builds the relativePose/fundamentalMatrix/world-to-cam part of the
        // context; baseline/medianDepth are supplied by the caller (already
        // computed for the baseline-gate check) rather than rederived here.
        KeyframePairContext buildPairContext(
            const Keyframe &current, const Keyframe &neighbor, double baseline, double medianDepth) const;

        std::vector<CandidateCorrespondence> findCandidateCorrespondences(const KeyframePairContext &context) const;
        TriangulationOutcome triangulateCorrespondence(
            const CandidateCorrespondence &correspondence, const KeyframePairContext &context);
        MapPointId insertMapPoint(const cv::Point3d &position, const CandidateCorrespondence &correspondence);

        // True if point (already projected into keyframe as `projection`,
        // by the caller -- shared with findProjectionMatches() so the
        // projection itself is only computed once) is in front of the
        // camera, inside the image bounds, within point's established
        // observed-distance range, and not viewed from too oblique an
        // angle relative to point's normal. The distance/angle checks are
        // skipped while those fields are still unestablished (fresh
        // points).
        bool isVisible(const MapPoint &point, const Keyframe &keyframe, const geometry::ProjectionResult &projection) const;

        // The observation whose descriptor has the smallest median Hamming
        // distance to all the other observations' descriptors -- more
        // robust than just picking the first observation, since any single
        // sighting can be blurry or seen at an oblique angle. Returns
        // nullptr if every KeyFrame that observed it is gone.
        const vision::OrbDescriptor *representativeDescriptor(const MapPoint &point) const;

        // Keypoint indices in `keyframe` within the projection search
        // window of `projection`'s image point, passing the Hamming
        // distance descriptor threshold -- sorted best (lowest distance)
        // first, with an ambiguity check (best-vs-second-best ratio) that
        // discards the whole match if the two closest candidates are too
        // close to call. Empty if nothing nearby matches well enough or the
        // best two matches are too ambiguous.
        std::vector<size_t> findProjectionMatches(
            const MapPoint &point, const Keyframe &keyframe, const geometry::ProjectionResult &projection) const;

        // The MapPoint to keep when two MapPoints turn out to be duplicate
        // observations of the same physical point: more observations wins;
        // ties broken by lower id (the older point).
        MapPointId chooseSurvivor(MapPointId lhs, MapPointId rhs) const;

        // Moves every observation from `duplicate` onto `survivor`
        // (skipping KeyFrames that already see `survivor` through a
        // different feature) and erases `duplicate` from the Map.
        void mergeMapPoints(MapPointId survivor, MapPointId duplicate);

        // Rebuilds the CovisibilityGraph once after a batch of fusions --
        // the Observation Graph itself is kept consistent incrementally by
        // mergeMapPoints()/Map::addObservation() as each fusion happens.
        void updateGraphsAfterFusion();

        Mapping &mapping_;

        vision::DescriptorMatcher matcher_;
        geometry::Triangulator triangulator_;

        std::queue<KeyframeId> keyFrameQueue_;
    };
}
