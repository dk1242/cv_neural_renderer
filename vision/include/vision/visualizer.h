#pragma once

#include <opencv2/opencv.hpp>

namespace vision
{

    class Visualizer
    {
        int screenCnt = 0;

    public:
        static cv::Mat normalizeToDisplay(
            const cv::Mat &image);
        void display(const cv::Mat &image, const std::string &windowName);
    };

}