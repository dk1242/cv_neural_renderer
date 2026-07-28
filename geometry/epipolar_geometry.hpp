#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace geometry
{

    struct CameraPose
    {
        cv::Mat R; // 3x3 rotation
        cv::Mat t; // 3x1 translation
    };

    struct ProjectionResult
    {
        cv::Point2d imagePoint;
        bool visible;
    };

    struct EpipolarLine
    {
        double a;
        double b;
        double c;
    };

    class EpipolarGeometry
    {
    public:
        static ProjectionResult projectPoint(const cv::Point3d &pointWorld, const CameraPose &pose, const cv::Mat &K);

        static cv::Point3d cameraCenter(const CameraPose &pose);

        static cv::Point2d computeEpipole(const CameraPose &source, const CameraPose &target, const cv::Mat &K);

        static double parallaxAngle(const cv::Point3d &pointWorld, const CameraPose &camera1, const CameraPose &camera2);

        // The epipolar line in the second image that a point in the first
        // image (given the two views' fundamental matrix) must lie on: l = F * x.
        static EpipolarLine computeEpipolarLine(const cv::Mat &F, const cv::Point2d &point);

        // Perpendicular distance from point to line, in pixels.
        static double distanceToLine(const EpipolarLine &line, const cv::Point2d &point);
    };

}