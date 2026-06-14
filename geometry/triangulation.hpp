#pragma once

#include <opencv2/opencv.hpp>

namespace geometry
{

    class Triangulator
    {
    public:
        cv::Point3d triangulatePoint(const cv::Point2d &x1, const cv::Point2d &x2,
                                     const cv::Mat &P1, const cv::Mat &P2);
    };

}
