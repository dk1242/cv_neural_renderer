#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace vision
{

class LucasKanadeTracker
{
public:
    std::vector<cv::Point2f> track(const cv::Mat& previous,
                                   const cv::Mat& current,
                                   const std::vector<cv::Point2f>& points);
};

}
