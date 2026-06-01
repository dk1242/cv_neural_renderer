#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace vision
{
    struct KeyPoint
    {
        cv::Point position;
        float score;
    };

    const std::array<cv::Point, 16> circle = {
        cv::Point(0, -3),
        cv::Point(1, -3),
        cv::Point(2, -2),
        cv::Point(3, -1),
        cv::Point(3, 0),
        cv::Point(3, 1),
        cv::Point(2, 2),
        cv::Point(1, 3),
        cv::Point(0, 3),
        cv::Point(-1, 3),
        cv::Point(-2, 2),
        cv::Point(-3, 1),
        cv::Point(-3, 0),
        cv::Point(-3, -1),
        cv::Point(-2, -2),
        cv::Point(-1, -3)};
    class FastCornerDetector
    {
        int m_threshold = 20;
        bool passesHighSpeedTest(const cv::Mat &image, int x, int y) const;
        bool isCorner(const cv::Mat &image, int x, int y) const;
        int computeScore(const cv::Mat &image, int x, int y) const;
        std::vector<KeyPoint> nonMaximumSuppression(const cv::Mat &scoreMap) const;

    public:
        std::vector<KeyPoint> detect(const cv::Mat &image);

        cv::Mat visualizeCorners(const cv::Mat &image,
                                 const std::vector<KeyPoint> &corners);
    };
}