#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <iostream>
#include "normalization.h"

namespace geometry
{

    class FundamentalMatrix
    {
    public:
        static cv::Mat estimate8Point(const std::vector<cv::Point2d> &points1, const std::vector<cv::Point2d> &points2);

    private:
        static cv::Mat buildAMatrix(const std::vector<cv::Point2d> &points1, const std::vector<cv::Point2d> &points2);

        static cv::Mat enforceRank2(const cv::Mat &F);
    };

}