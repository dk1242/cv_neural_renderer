#pragma once

#include <opencv2/opencv.hpp>
#include "epipolar_geometry.hpp"

namespace geometry
{

    class EssentialMatrixEstimator
    {
    public:
        cv::Mat computeEssentialMatrix(const cv::Mat &F, const cv::Mat &K);

        cv::Mat enforceEssentialConstraints(const cv::Mat &E);

        cv::Mat estimate(const cv::Mat &points1,
                         const cv::Mat &points2,
                         const cv::Mat &cameraMatrix);
    };

}
