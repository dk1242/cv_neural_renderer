#pragma once
#include <opencv2/opencv.hpp>

namespace geometry
{
    struct Correspondence
    {
        cv::Point2f source;
        cv::Point2f destination;
    };
}