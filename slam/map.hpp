#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace slam
{

class Map
{
public:
    void addPoint(const cv::Point3f& point);
    const std::vector<cv::Point3f>& points() const;

private:
    std::vector<cv::Point3f> points_;
};

}
