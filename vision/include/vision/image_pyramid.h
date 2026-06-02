#pragma once

#include <opencv2/opencv.hpp>
#include "vision/gaussian.hpp"
#include <vector>

namespace vision
{
    struct Level
    {
        cv::Mat image;
        float scale;
    };

    class ImagePyramid
    {
        cv::Mat downsample(const cv::Mat &image) const;

    public:
        std::vector<Level> build(const cv::Mat &image, int numLevels, float scaleFactor) const;
    };
}