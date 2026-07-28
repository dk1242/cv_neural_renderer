#pragma once

#include <optional>
#include <queue>

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

        Mapping &mapping_;

        vision::DescriptorMatcher matcher_;
        geometry::Triangulator triangulator_;

        std::queue<KeyframeId> keyFrameQueue_;
    };
}
