#pragma once

#include <opencv2/opencv.hpp>
#include "epipolar_geometry.hpp"
#include "triangulation.hpp"

namespace geometry
{

    class PoseRecovery
    {
    public:
        geometry::Triangulator m_triangulator;
        int countPositiveDepth(const geometry::CameraPose &pose, const std::vector<cv::Point2d> &points1,
                               const std::vector<cv::Point2d> &points2, const cv::Mat &K);
        std::vector<geometry::CameraPose> decomposeEssentialMatrix(const cv::Mat &E);
        geometry::CameraPose recoverPose(const cv::Mat &E,
                                         const std::vector<cv::Point2d> &points1,
                                         const std::vector<cv::Point2d> &points2,
                                         const cv::Mat &K);
    };
}