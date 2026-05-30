#pragma once

#include <opencv2/opencv.hpp>

namespace geometry
{

class RansacEstimator
{
public:
    cv::Mat estimate(const cv::Mat& sourcePoints,
                     const cv::Mat& destinationPoints);
};

}
