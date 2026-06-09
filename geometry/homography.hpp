#pragma once

#include <opencv2/opencv.hpp>
#include "correspondence.hpp"

namespace geometry
{

    class HomographyEstimator
    {
    public:
        cv::Point2f projectPoint(const cv::Point2f &point, const cv::Mat &H) const;
        float reprojectionError(const geometry::Correspondence &correspondence, const cv::Mat &H) const;
        cv::Mat estimate(const std::vector<geometry::Correspondence> &correspondences);
    };

}
