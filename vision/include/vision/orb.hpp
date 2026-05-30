#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace vision
{

class OrbExtractor
{
public:
    void compute(const cv::Mat& image,
                 std::vector<cv::KeyPoint>& keypoints,
                 cv::Mat& descriptors);
};

}
