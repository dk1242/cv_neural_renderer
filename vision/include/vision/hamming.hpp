#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace vision
{

class HammingMatcher
{
public:
    std::vector<cv::DMatch> match(const cv::Mat& queryDescriptors,
                                  const cv::Mat& trainDescriptors);
};

}
