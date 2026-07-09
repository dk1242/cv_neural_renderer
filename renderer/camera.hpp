#pragma once

#include <opencv2/opencv.hpp>

namespace renderer
{

    class Camera
    {
    public:
        void setPose(const cv::Mat &pose);
        const cv::Mat &pose() const;

    private:
        cv::Mat pose_;
    };
}
