#include "slam/localmapping.hpp"

#include <algorithm>
#include <limits>
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

constexpr double kFusionSearchRadiusPx = 5.0;                // fixed window; ORB-SLAM scales this by keypoint octave, which we don't track yet
constexpr int kMaxHammingDistance = 50;                       // ORB-SLAM's TH_LOW, matches DescriptorMatcher::match()'s own threshold
constexpr double kMinViewingCosine = 0.5;                     // ORB-SLAM's IsInFrustum gate: reject views >~60 degrees off the point's normal

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

    fuseMapPoints(id, localMapPoints);

    // later
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

bool slam::LocalMapping::isVisible(const MapPoint &point, const Keyframe &keyframe, const geometry::ProjectionResult &projection) const
{
    if (!projection.visible)
    {
        return false;
    }

    // cameraMatrix() is built with the principal point at the image center
    // (see VisualOdometry::buildCameraMatrix), so 2*cx/2*cy recovers the
    // image bounds without Keyframe having to store width/height itself.
    const cv::Mat &K = mapping_.cameraMatrix();
    const double width = 2.0 * K.at<double>(0, 2);
    const double height = 2.0 * K.at<double>(1, 2);

    if (projection.imagePoint.x < 0.0 || projection.imagePoint.x >= width ||
        projection.imagePoint.y < 0.0 || projection.imagePoint.y >= height)
    {
        return false;
    }

    const cv::Mat &twc = keyframe.pose().twc;
    const cv::Point3d cameraCenter(twc.at<double>(0), twc.at<double>(1), twc.at<double>(2));

    cv::Point3d viewDirection = point.position - cameraCenter;
    const double distance = cv::norm(viewDirection);
    if (distance <= 0.0)
    {
        return false;
    }

    // Unestablished (0.0/0.0) means "not enough observations yet" -- skip
    // the range gate rather than reject a fresh point outright.
    if (point.minDistance > 0.0 && (distance < point.minDistance || distance > point.maxDistance))
    {
        return false;
    }

    static const cv::Point3d kZeroNormal(0.0, 0.0, 0.0);
    if (point.normal == kZeroNormal)
    {
        // No established viewing direction yet (fresh/single-observation
        // point) -- nothing to gate against, so let it through.
        return true;
    }

    viewDirection *= (1.0 / distance);

    // Descriptors become unreliable once the viewpoint has rotated too far
    // from the directions the point was originally seen from -- ORB-SLAM
    // rejects past ~60 degrees (cos < 0.5).
    return viewDirection.dot(point.normal) >= kMinViewingCosine;
}

const vision::OrbDescriptor *slam::LocalMapping::representativeDescriptor(const MapPoint &point) const
{
    std::vector<const vision::OrbDescriptor *> descriptors;
    descriptors.reserve(point.numObservations());

    for (const auto &[keyframeId, observation] : point.observations())
    {
        const Keyframe *keyframe = mapping_.map().getKeyframe(keyframeId);
        if (keyframe && observation.keypointIndex < keyframe->descriptors().size())
        {
            descriptors.push_back(&keyframe->descriptors()[observation.keypointIndex]);
        }
    }

    if (descriptors.empty())
    {
        return nullptr;
    }
    if (descriptors.size() == 1)
    {
        return descriptors.front();
    }

    // The descriptor whose median Hamming distance to all the others is
    // smallest -- more robust than just picking the first observation,
    // since any single sighting can be blurry or seen at an oblique angle.
    // O(n^2) in observation count, same as ORB-SLAM's own
    // ComputeDistinctiveDescriptors(); n stays small (a handful of
    // KeyFrames) so this is cheap in practice.
    size_t bestIndex = 0;
    int bestMedianDistance = std::numeric_limits<int>::max();

    for (size_t i = 0; i < descriptors.size(); ++i)
    {
        std::vector<int> distances;
        distances.reserve(descriptors.size() - 1);
        for (size_t j = 0; j < descriptors.size(); ++j)
        {
            if (i == j)
            {
                continue;
            }
            distances.push_back(matcher_.hammingDistance(*descriptors[i], *descriptors[j]));
        }

        const auto middle = distances.begin() + static_cast<std::ptrdiff_t>(distances.size() / 2);
        std::nth_element(distances.begin(), middle, distances.end());
        const int medianDistance = *middle;

        if (medianDistance < bestMedianDistance)
        {
            bestMedianDistance = medianDistance;
            bestIndex = i;
        }
    }

    return descriptors[bestIndex];
}

std::vector<size_t> slam::LocalMapping::findProjectionMatches(
    const MapPoint &point, const Keyframe &keyframe, const geometry::ProjectionResult &projection) const
{
    std::vector<size_t> matches;

    if (!projection.visible)
    {
        return matches;
    }

    const vision::OrbDescriptor *descriptor = representativeDescriptor(point);
    if (!descriptor)
    {
        return matches;
    }

    std::vector<std::pair<size_t, int>> candidates;
    const auto &keypoints = keyframe.keypoints();
    const auto &descriptors = keyframe.descriptors();

    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        const cv::Point2d keypointPixel(keypoints[i].position.x, keypoints[i].position.y);
        if (cv::norm(keypointPixel - projection.imagePoint) > kFusionSearchRadiusPx)
        {
            continue;
        }

        const int distance = matcher_.hammingDistance(*descriptor, descriptors[i]);
        if (distance > kMaxHammingDistance)
        {
            continue;
        }

        candidates.emplace_back(i, distance);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto &a, const auto &b) { return a.second < b.second; });

    // Ambiguity check: if the two closest candidates are nearly tied, we
    // can't tell which one is the real match -- discard rather than guess
    // and risk fusing against the wrong feature.
    if (candidates.size() >= 2)
    {
        const float ratio = static_cast<float>(candidates[0].second) / static_cast<float>(candidates[1].second);
        if (ratio >= kRatioTestThreshold)
        {
            return matches;
        }
    }

    matches.reserve(candidates.size());
    for (const auto &[keypointIndex, distance] : candidates)
    {
        matches.push_back(keypointIndex);
    }

    return matches;
}

slam::MapPointId slam::LocalMapping::chooseSurvivor(MapPointId lhs, MapPointId rhs) const
{
    const MapPoint *lhsPoint = mapping_.map().find(lhs);
    const MapPoint *rhsPoint = mapping_.map().find(rhs);

    if (!lhsPoint)
    {
        return rhs;
    }
    if (!rhsPoint)
    {
        return lhs;
    }

    if (lhsPoint->numObservations() != rhsPoint->numObservations())
    {
        return lhsPoint->numObservations() > rhsPoint->numObservations() ? lhs : rhs;
    }

    return std::min(lhs, rhs);
}

void slam::LocalMapping::mergeMapPoints(MapPointId survivor, MapPointId duplicate)
{
    if (survivor == duplicate)
    {
        return;
    }

    const MapPoint *duplicatePoint = mapping_.map().find(duplicate);
    const MapPoint *survivorPoint = mapping_.map().find(survivor);
    if (!duplicatePoint || !survivorPoint)
    {
        return;
    }

    // Snapshot -- Map::removeObservation() below mutates duplicatePoint's
    // own observation map, which we can't iterate while modifying.
    const auto observations = duplicatePoint->observations();

    for (const auto &[keyframeId, observation] : observations)
    {
        if (!survivorPoint->hasObservation(keyframeId))
        {
            mapping_.map().addObservation(survivor, observation);
        }
        mapping_.map().removeObservation(duplicate, keyframeId);
    }

    mapping_.map().remove(duplicate);
}

void slam::LocalMapping::updateGraphsAfterFusion()
{
    mapping_.refreshCovisibilityGraph();
}

void slam::LocalMapping::fuseMapPoints(KeyframeId currentKeyframeId, const LocalMapPointSet &localMapPoints)
{
    const Keyframe *currentKeyframe = mapping_.map().getKeyframe(currentKeyframeId);
    if (!currentKeyframe)
    {
        return;
    }

    const auto poseWorldToCam = worldToCameraPose(currentKeyframe->pose());

    std::unordered_map<size_t, MapPointId> keypointToMapPoint;
    for (const auto &[existingMapPointId, observation] : currentKeyframe->observations())
    {
        keypointToMapPoint.emplace(observation.keypointIndex, existingMapPointId);
    }

    bool fusedAny = false;

    for (const auto &mapPointId : localMapPoints.all)
    {
        const MapPoint *point = mapping_.map().find(mapPointId);
        if (!point || !point->active || point->hasObservation(currentKeyframeId))
        {
            continue;
        }

        // Projected once and shared between isVisible() and
        // findProjectionMatches() rather than each recomputing it.
        const auto projection = geometry::EpipolarGeometry::projectPoint(point->position, poseWorldToCam, mapping_.cameraMatrix());

        if (!isVisible(*point, *currentKeyframe, projection))
        {
            continue;
        }

        const auto matches = findProjectionMatches(*point, *currentKeyframe, projection);
        if (matches.empty())
        {
            continue;
        }

        const size_t keypointIndex = matches.front();
        const auto existingIt = keypointToMapPoint.find(keypointIndex);

        if (existingIt == keypointToMapPoint.end())
        {
            const cv::Point2f &pixel = currentKeyframe->keypoints()[keypointIndex].position;
            mapping_.map().addObservation(mapPointId, MapObservation{currentKeyframeId, keypointIndex, pixel});
            keypointToMapPoint.emplace(keypointIndex, mapPointId);
            fusedAny = true;
            continue;
        }

        if (existingIt->second == mapPointId)
        {
            continue;
        }

        const MapPointId survivor = chooseSurvivor(mapPointId, existingIt->second);
        const MapPointId duplicate = (survivor == mapPointId) ? existingIt->second : mapPointId;
        mergeMapPoints(survivor, duplicate);
        keypointToMapPoint[keypointIndex] = survivor;
        fusedAny = true;
    }

    if (fusedAny)
    {
        updateGraphsAfterFusion();
    }
}