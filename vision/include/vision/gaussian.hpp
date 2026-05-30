#pragma once

#include <opencv2/opencv.hpp>

namespace vision
{

class GaussianFilter
{
public:
    cv::Mat apply(const cv::Mat& image);
};

}
