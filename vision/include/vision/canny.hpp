#pragma once

#include <opencv2/opencv.hpp>

namespace vision
{

class CannyDetector
{
public:
    cv::Mat detect(const cv::Mat& image);
};

}
