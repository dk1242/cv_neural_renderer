#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace vision
{

class FastDetector
{
public:
    std::vector<cv::KeyPoint> detect(const cv::Mat& image);
};

}
