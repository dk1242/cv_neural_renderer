#pragma once

#include <opencv2/core.hpp>
#include <vector>

namespace geometry
{

    struct NormalizationResult
    {
        std::vector<cv::Point2d> points;
        cv::Mat T;
    };
    class Normalization
    {
    public:
        NormalizationResult normalizePoints(const std::vector<cv::Point2d> &points);
    };
}