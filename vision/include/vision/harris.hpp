#pragma once

#include <opencv2/opencv.hpp>

namespace vision
{

class HarrisDetector
{
public:
    cv::Mat detect(const cv::Mat& image);
};

}
