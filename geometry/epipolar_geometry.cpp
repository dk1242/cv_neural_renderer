#include "epipolar_geometry.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

geometry::ProjectionResult geometry::EpipolarGeometry::projectPoint(const cv::Point3d &pointWorld,
                                                                    const CameraPose &pose, const cv::Mat &K)
{
    geometry::ProjectionResult result;
    cv::Mat X = (cv::Mat_<double>(3, 1) << pointWorld.x, pointWorld.y, pointWorld.z);
    cv::Mat Xc = pose.R * X + pose.t;

    double Z = Xc.at<double>(2);
    if (Z <= 0.0)
    {
        result.visible = false;
        return result;
    }

    cv::Mat x = K * Xc;
    double u = x.at<double>(0);
    double v = x.at<double>(1);
    double w = x.at<double>(2);

    result.imagePoint = cv::Point2d(u / w, v / w);
    result.visible = true;
    return result;
}

cv::Point3d geometry::EpipolarGeometry::cameraCenter(const CameraPose &pose)
{
    // cv::Mat R;
    // cv::invert(pose.R, R);
    // cv::Mat t = -R * pose.t;
    // return cv::Point3d(t.at<double>(0), t.at<double>(1), t.at<double>(2));

    cv::Mat C = -pose.R.t() * pose.t;
    return cv::Point3d(C.at<double>(0), C.at<double>(1), C.at<double>(2));
}

cv::Point2d geometry::EpipolarGeometry::computeEpipole(const CameraPose &source, const CameraPose &target, const cv::Mat &K)
{
    // cv::Mat E = target.R * source.R.t();
    // cv::Mat t = target.t - E * source.t;
    // cv::Mat e = K * t;
    // return cv::Point2d(e.at<double>(0) / e.at<double>(2), e.at<double>(1) / e.at<double>(2));

    cv::Point3d C1 = cameraCenter(source);
    ProjectionResult result = projectPoint(C1, target, K);

    if (!result.visible)
    {
        std::cout << "Epipole lies at infinity\n";
    }
    return result.imagePoint;
}

double geometry::EpipolarGeometry::parallaxAngle(const cv::Point3d &pointWorld, const CameraPose &camera1, const CameraPose &camera2)
{
    cv::Mat X = (cv::Mat_<double>(3, 1) << pointWorld.x, pointWorld.y, pointWorld.z);
    cv::Point3d C1 = cameraCenter(camera1);
    cv::Point3d C2 = cameraCenter(camera2);

    cv::Mat Xc1 = (cv::Mat_<double>(3, 1) << C1.x, C1.y, C1.z);
    Xc1 = X - Xc1;

    cv::Mat Xc2 = (cv::Mat_<double>(3, 1) << C2.x, C2.y, C2.z);
    Xc2 = X - Xc2;

    double dotProduct = Xc1.dot(Xc2);
    double norm1 = cv::norm(Xc1);
    double norm2 = cv::norm(Xc2);

    double cosParallax = std::clamp(dotProduct / (norm1 * norm2), -1.0, 1.0);
    return std::acos(cosParallax);
}
