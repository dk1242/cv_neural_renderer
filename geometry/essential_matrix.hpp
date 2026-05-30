#pragma once

#include <opencv2/opencv.hpp>

namespace geometry
{

class EssentialMatrixEstimator
{
public:
    cv::Mat estimate(const cv::Mat& points1,
                     const cv::Mat& points2,
                     const cv::Mat& cameraMatrix);
};

}
