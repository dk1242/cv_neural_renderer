#pragma once

#include <opencv2/opencv.hpp>
#include "gaussian.hpp"
#include "sobel.hpp"

namespace vision
{
    struct CannyResult
    {
        cv::Mat blurred;

        cv::Mat gx;
        cv::Mat gy;

        cv::Mat magnitude;
        cv::Mat direction;

        cv::Mat nms;

        cv::Mat thresholded;

        cv::Mat edges;
    };
    class CannyDetector
    {
    public:
        cv::Mat detectEdges(const cv::Mat &image, float lowThreshold,
                            float highThreshold);

        CannyResult detectDebug(const cv::Mat &image, float lowThreshold,
                                float highThreshold);

    private:
        vision::GaussianFilter m_gaussianFilter;
        vision::SobelFilter m_sobelFilter;

        cv::Mat nonMaximumSuppression(const cv::Mat &magnitude,
                                      const cv::Mat &angle);

        cv::Mat doubleThreshold(const cv::Mat &nms, float lowThreshold,
                                float highThreshold);

        cv::Mat hysteresis(
            const cv::Mat &thresholded);
    };

}
