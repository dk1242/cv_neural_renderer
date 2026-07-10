#include "slam/mapping.hpp"

slam::Mapping::UpdateStats slam::Mapping::update(TrackManager &trackManager, FrameId currentFrameId,
                                                 const geometry::CameraPose &relativePose,
                                                 const CameraPose &previousWorldPose,
                                                 const cv::Mat &cameraMatrix)
{
    UpdateStats stats;

    if (relativePose.R.empty() || relativePose.t.empty() ||
        previousWorldPose.Rwc.empty() || previousWorldPose.twc.empty() ||
        cameraMatrix.empty())
    {
        return stats;
    }

    const geometry::CameraPose identityPose{cv::Mat::eye(3, 3, CV_64F), cv::Mat::zeros(3, 1, CV_64F)};

    for (const auto &[trackId, track] : trackManager.tracks())
    {
        if (!track.active() || track.mapPointId() ||
            track.observations().size() < 4 ||
            track.lastObservation().frameId != currentFrameId)
        {
            continue;
        }

        // TODO:
        // Don't always triangulate using the last two observations.
        // Choose the observation pair with the largest baseline/parallax
        // once KeyFrames are introduced.
        const auto &previousObservation = track.observations()[track.observations().size() - 2];
        const auto &currentObservation = track.lastObservation();

        const auto worldPoint = triangulateWorldPoint(
            cv::Point2d(previousObservation.pixel), cv::Point2d(currentObservation.pixel),
            identityPose, relativePose, cameraMatrix, previousWorldPose);
        if (!worldPoint)
        {
            continue;
        }

        const MapPointId mapPointId = map_.createPoint(*worldPoint, track.observations().size());
        trackManager.linkTrackToMapPoint(trackId, mapPointId);

        ++stats.newMapPoints;
        stats.promotions.emplace_back(trackId, mapPointId);
        // std::cout
        //     << "Track " << track.id()
        //     << " observations = "
        //     << track.observations().size()
        //     << '\n';
    }

    return stats;
}

std::optional<cv::Point3d> slam::Mapping::triangulateWorldPoint(
    const cv::Point2d &pixelPrev, const cv::Point2d &pixelCurr,
    const geometry::CameraPose &pose1, const geometry::CameraPose &pose2,
    const cv::Mat &cameraMatrix, const CameraPose &previousWorldPose)
{
    const cv::Point3d pointCamera = triangulator_.triangulatePoint(pixelPrev, pixelCurr, pose1, pose2, cameraMatrix);
    if (pointCamera.z <= 0.0)
    {
        return std::nullopt;
    }

    const cv::Mat Xc = (cv::Mat_<double>(3, 1) << pointCamera.x, pointCamera.y, pointCamera.z);
    const cv::Mat Xw = previousWorldPose.Rwc * Xc + previousWorldPose.twc;

    return cv::Point3d{Xw.at<double>(0), Xw.at<double>(1), Xw.at<double>(2)};
}

const slam::Map &slam::Mapping::map() const
{
    return map_;
}
