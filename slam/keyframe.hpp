#pragma once

#include <opencv2/opencv.hpp>

namespace slam
{

class Keyframe
{
public:
    explicit Keyframe(const cv::Mat& image);

    const cv::Mat& image() const;

private:
    cv::Mat image_;
};

}
