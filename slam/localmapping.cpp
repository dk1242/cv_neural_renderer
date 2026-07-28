#include "slam/localmapping.hpp"

#include <algorithm>
#include <unordered_set>

#include "geometry/epipolar_geometry.hpp"
#include "geometry/fundamental_matrix.h"
#include "slam/map.hpp"

constexpr size_t kMinCovisibilityWeight = 5;

constexpr float kRatioTestThreshold = 0.75f;                // matches VisualOdometry's own ratio test threshold
constexpr double kEpipolarDistancePx = 3.0;                  // matches RansacFundamental's default inlier threshold
constexpr double kReprojectionThresholdPx = 3.0;             // ditto
constexpr double kMinBaselineDepthRatio = 0.01;              // ORB-SLAM's baseline/scene-depth gate
constexpr double kMinParallaxRad = 0.0174533;                // ~1 degree

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
    const auto triangulationStats = triangulateNewMapPoints(id, localKeyframes);

    std::cout << "Processing KF " << id
              << " -- local KFs: " << localKeyframes.all.size()
              << ", local MapPoints: " << localMapPoints.all.size()
              << ", new MapPoints: " << triangulationStats.insertedMapPoints << '\n';

    // later
    // fuseMapPoints(...)
    // localBundleAdjustment(...)
    // keyFrameCulling(...)
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

geometry::CameraPose slam::LocalMapping::computeRelativePose(const Keyframe &from, const Keyframe &to) const
{
    const cv::Mat &RwcFrom = from.pose().Rwc;
    const cv::Mat &twcFrom = from.pose().twc;
    const cv::Mat &RwcTo = to.pose().Rwc;
    const cv::Mat &twcTo = to.pose().twc;

    geometry::CameraPose relative;
    relative.R = RwcTo.t() * RwcFrom;
    relative.t = RwcTo.t() * (twcFrom - twcTo);
    return relative;
}

geometry::CameraPose slam::LocalMapping::worldToCameraPose(const CameraPose &pose) const
{
    geometry::CameraPose result;
    result.R = pose.Rwc.t();
    result.t = -pose.Rwc.t() * pose.twc;
    return result;
}

double slam::LocalMapping::computeMedianSceneDepth(const Keyframe &keyframe) const
{
    const auto poseWorldToCam = worldToCameraPose(keyframe.pose());

    std::vector<double> depths;
    depths.reserve(keyframe.numObservations());

    for (const auto &[mapPointId, observation] : keyframe.observations())
    {
        const MapPoint *point = mapping_.map().find(mapPointId);
        if (!point)
        {
            continue;
        }

        const cv::Mat Xw = (cv::Mat_<double>(3, 1) << point->position.x, point->position.y, point->position.z);
        const cv::Mat Xc = poseWorldToCam.R * Xw + poseWorldToCam.t;
        depths.push_back(Xc.at<double>(2));
    }

    if (depths.empty())
    {
        return 0.0;
    }

    const auto middle = depths.begin() + static_cast<std::ptrdiff_t>(depths.size() / 2);
    std::nth_element(depths.begin(), middle, depths.end());
    return *middle;
}

std::vector<slam::CandidateCorrespondence> slam::LocalMapping::findCandidateCorrespondences(
    const KeyframePairContext &context) const
{
    const Keyframe &current = context.current;
    const Keyframe &neighbor = context.neighbor;

    std::vector<CandidateCorrespondence> candidates;

    std::unordered_set<size_t> currentHasMapPoint;
    for (const auto &[mapPointId, observation] : current.observations())
    {
        currentHasMapPoint.insert(observation.keypointIndex);
    }

    std::unordered_set<size_t> neighborHasMapPoint;
    for (const auto &[mapPointId, observation] : neighbor.observations())
    {
        neighborHasMapPoint.insert(observation.keypointIndex);
    }

    const auto forwardKnn = matcher_.knnMatch(current.descriptors(), neighbor.descriptors());
    const auto forwardMatches = matcher_.ratioTest(forwardKnn, kRatioTestThreshold);
    const auto backwardKnn = matcher_.knnMatch(neighbor.descriptors(), current.descriptors());
    const auto backwardMatches = matcher_.ratioTest(backwardKnn, kRatioTestThreshold);
    const auto crossMatches = matcher_.crosscheckMatch(forwardMatches, backwardMatches);

    for (const auto &match : crossMatches)
    {
        const auto currentIndex = static_cast<size_t>(match.queryIdx);
        const auto neighborIndex = static_cast<size_t>(match.trainIdx);

        if (currentHasMapPoint.count(currentIndex) || neighborHasMapPoint.count(neighborIndex))
        {
            continue;
        }

        const cv::Point2d currentPixel(current.keypoints()[currentIndex].position.x,
                                       current.keypoints()[currentIndex].position.y);
        const cv::Point2d neighborPixel(neighbor.keypoints()[neighborIndex].position.x,
                                        neighbor.keypoints()[neighborIndex].position.y);

        const auto line = geometry::EpipolarGeometry::computeEpipolarLine(context.fundamentalMatrix, currentPixel);
        if (geometry::EpipolarGeometry::distanceToLine(line, neighborPixel) > kEpipolarDistancePx)
        {
            continue;
        }

        candidates.push_back(CandidateCorrespondence{
            current.id(), neighbor.id(),
            currentIndex, neighborIndex,
            currentPixel, neighborPixel});
    }

    return candidates;
}

slam::TriangulationOutcome slam::LocalMapping::triangulateCorrespondence(
    const CandidateCorrespondence &correspondence, const KeyframePairContext &context)
{
    const geometry::CameraPose identityPose{cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F)};

    const cv::Point3d Xc = triangulator_.triangulatePoint(
        correspondence.currentPixel, correspondence.neighborPixel,
        identityPose, context.relativePose, mapping_.cameraMatrix());

    const cv::Mat XcMat = (cv::Mat_<double>(3, 1) << Xc.x, Xc.y, Xc.z);
    const cv::Mat XwMat = context.current.pose().Rwc * XcMat + context.current.pose().twc;
    const cv::Point3d Xw(XwMat.at<double>(0), XwMat.at<double>(1), XwMat.at<double>(2));

    const double parallax = geometry::EpipolarGeometry::parallaxAngle(
        Xw, context.poseWorldToCamCurrent, context.poseWorldToCamNeighbor);
    if (parallax < kMinParallaxRad)
    {
        return TriangulationOutcome{std::nullopt, TriangulationRejection::kParallax};
    }

    if (Xc.z <= 0.0)
    {
        return TriangulationOutcome{std::nullopt, TriangulationRejection::kCheirality};
    }

    const cv::Mat XcNeighbor = context.relativePose.R * XcMat + context.relativePose.t;
    if (XcNeighbor.at<double>(2) <= 0.0)
    {
        return TriangulationOutcome{std::nullopt, TriangulationRejection::kCheirality};
    }

    const auto projCurrent = geometry::EpipolarGeometry::projectPoint(Xw, context.poseWorldToCamCurrent, mapping_.cameraMatrix());
    const auto projNeighbor = geometry::EpipolarGeometry::projectPoint(Xw, context.poseWorldToCamNeighbor, mapping_.cameraMatrix());

    if (!projCurrent.visible || !projNeighbor.visible ||
        cv::norm(projCurrent.imagePoint - correspondence.currentPixel) > kReprojectionThresholdPx ||
        cv::norm(projNeighbor.imagePoint - correspondence.neighborPixel) > kReprojectionThresholdPx)
    {
        return TriangulationOutcome{std::nullopt, TriangulationRejection::kReprojection};
    }

    return TriangulationOutcome{Xw, TriangulationRejection::kCheirality};
}

slam::KeyframePairContext slam::LocalMapping::buildPairContext(
    const Keyframe &current, const Keyframe &neighbor, double baseline, double medianDepth) const
{
    KeyframePairContext context{current, neighbor};

    context.relativePose = computeRelativePose(current, neighbor);
    context.fundamentalMatrix = geometry::FundamentalMatrix{}.computeFundamentalGroundTruth(
        mapping_.cameraMatrix(), context.relativePose.R, context.relativePose.t);

    context.poseWorldToCamCurrent = worldToCameraPose(current.pose());
    context.poseWorldToCamNeighbor = worldToCameraPose(neighbor.pose());

    context.baseline = baseline;
    context.medianDepth = medianDepth;

    return context;
}

slam::MapPointId slam::LocalMapping::insertMapPoint(const cv::Point3d &position, const CandidateCorrespondence &correspondence)
{
    const MapPointId mapPointId = mapping_.map().createPoint(position, 2);

    mapping_.map().addObservation(mapPointId, MapObservation{
        correspondence.currentKeyframeId, correspondence.currentKeypointIndex, correspondence.currentPixel});
    mapping_.map().addObservation(mapPointId, MapObservation{
        correspondence.neighborKeyframeId, correspondence.neighborKeypointIndex, correspondence.neighborPixel});

    return mapPointId;
}

slam::TriangulationStats slam::LocalMapping::triangulateNewMapPoints(KeyframeId current, const LocalKeyFrameSet &localKeyframes)
{
    TriangulationStats stats;

    const Keyframe *currentKeyframe = mapping_.map().getKeyframe(current);
    if (!currentKeyframe)
    {
        return stats;
    }

    for (const auto &neighborId : localKeyframes.firstOrder)
    {
        const Keyframe *neighborKeyframe = mapping_.map().getKeyframe(neighborId);
        if (!neighborKeyframe)
        {
            continue;
        }

        ++stats.candidatePairs;

        const double baseline = cv::norm(currentKeyframe->pose().twc - neighborKeyframe->pose().twc);
        const double medianDepth = computeMedianSceneDepth(*neighborKeyframe);

        if (medianDepth <= 0.0 || baseline / medianDepth <= kMinBaselineDepthRatio)
        {
            ++stats.rejectedBaseline;
            continue;
        }

        const auto context = buildPairContext(*currentKeyframe, *neighborKeyframe, baseline, medianDepth);
        const auto candidates = findCandidateCorrespondences(context);
        stats.candidateMatches += candidates.size();

        for (const auto &candidate : candidates)
        {
            const auto outcome = triangulateCorrespondence(candidate, context);
            if (!outcome.point)
            {
                if (outcome.rejection == TriangulationRejection::kParallax)
                {
                    ++stats.rejectedParallax;
                }
                else if (outcome.rejection == TriangulationRejection::kCheirality)
                {
                    ++stats.rejectedCheirality;
                }
                else
                {
                    ++stats.rejectedReprojection;
                }
                continue;
            }

            insertMapPoint(*outcome.point, candidate);
            ++stats.insertedMapPoints;
        }
    }

    if (stats.insertedMapPoints > 0)
    {
        mapping_.refreshCovisibilityGraph();
    }

    return stats;
}