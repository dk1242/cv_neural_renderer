#pragma once

#include <opencv2/opencv.hpp>

namespace slam
{

    class VisualOdometry
    {
    public:
        void processFrame(const cv::Mat &frame);
    };

}
