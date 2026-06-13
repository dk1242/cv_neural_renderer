#include "geometry/epipolar_geometry.hpp"
#include <iostream>
#include <numbers>

geometry::CameraPose createPose(
    const cv::Mat &R,
    const cv::Mat &cameraCenter)
{
    geometry::CameraPose pose;
    pose.R = R;
    cv::Mat C = cameraCenter;
    pose.t = -R * C;
    return pose;
}

int main()
{
    geometry::EpipolarGeometry epipolar;

    cv::Mat camera1 = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    cv::Mat camera2 = (cv::Mat_<double>(3, 1) << 1, 0, 0);

    double angle = 15.0 * CV_PI / 180.0;

    cv::Mat R =
        (cv::Mat_<double>(3, 3) << std::cos(angle), 0, std::sin(angle),
         0, 1, 0,
         -std::sin(angle), 0, std::cos(angle));
    geometry::CameraPose pose1 = createPose(cv::Mat::eye(3, 3, CV_64F), camera1);
    geometry::CameraPose pose2 = createPose(R, camera2);

    cv::Point3d Point1(0, 0, 5);
    cv::Point3d Point2(0, 0, 50);

    cv::Mat K = (cv::Mat_<double>(3, 3) << 800, 0, 320,
                 0, 800, 240,
                 0, 0, 1);

    cv::Point2d e12 = epipolar.computeEpipole(pose1, pose2, K);

    cv::Point2d e21 = epipolar.computeEpipole(pose2, pose1, K);

    std::cout << e12 << '\n';
    std::cout << e21 << '\n';
    return 0;
}
