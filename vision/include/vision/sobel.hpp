#pragma once

#include <opencv2/opencv.hpp>
#include "convolution.hpp"
namespace vision
{
    class SobelFilter
    {
        vision::ConvolutionFilter m_convolution;

    public:
        cv::Mat computeX(const cv::Mat &image);
        cv::Mat computeY(const cv::Mat &image);

        cv::Mat magnitude(
            const cv::Mat &gx,
            const cv::Mat &gy);

        cv::Mat direction(
            const cv::Mat &gx,
            const cv::Mat &gy);
    };

}
