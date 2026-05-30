#pragma once

#include <opencv2/opencv.hpp>

namespace vision
{

class SobelFilter
{
public:
    cv::Mat apply(const cv::Mat& image);
};

}
