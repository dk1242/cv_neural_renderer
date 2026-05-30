#pragma once

#include <opencv2/opencv.hpp>

namespace vision
{

class ConvolutionFilter
{
public:
    cv::Mat apply(const cv::Mat& image, const cv::Mat& kernel);
};

}
