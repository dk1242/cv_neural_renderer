#pragma once

#include <opencv2/opencv.hpp>

namespace geometry
{

class Triangulator
{
public:
    cv::Mat triangulate(const cv::Mat& projection1,
                        const cv::Mat& projection2,
                        const cv::Mat& points1,
                        const cv::Mat& points2);
};

}
