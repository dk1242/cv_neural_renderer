#pragma once

#include <opencv2/opencv.hpp>
#include "gaussian.hpp"
#include "sobel.hpp"

namespace vision
{

    class HarrisCornerDetector
    {
        vision::GaussianFilter m_gaussianFilter;
        vision::SobelFilter m_sobelFilter;
        float k = 0.04f;

    public:
        cv::Mat computeResponse(const cv::Mat &image);

        std::vector<cv::Point> detectCorners(const cv::Mat &image,
                                             float thresholdFactor);

        cv::Mat visualizeCorners(const cv::Mat &image,
                                 const std::vector<cv::Point> &corners);
    };

}
