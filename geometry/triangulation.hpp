#pragma once

#include <opencv2/opencv.hpp>

#include "epipolar_geometry.hpp"

namespace geometry
{

    class Triangulator
    {
    public:
        cv::Point3d triangulatePoint(const cv::Point2d &x1, const cv::Point2d &x2,
                                     const cv::Mat &P1, const cv::Mat &P2);

        // Builds P1 = K[R1|t1] and P2 = K[R2|t2] from the given camera poses
        // and triangulates -- projection matrix construction is geometry's
        // job, not a caller's.
        cv::Point3d triangulatePoint(const cv::Point2d &x1, const cv::Point2d &x2,
                                     const CameraPose &pose1, const CameraPose &pose2,
                                     const cv::Mat &K);
    };

}
