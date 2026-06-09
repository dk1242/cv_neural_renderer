#pragma once

#include <opencv2/opencv.hpp>
#include "correspondence.hpp"
#include "homography.hpp"
#include <random>

namespace geometry
{
    struct RansacResult
    {
        cv::Mat homography;

        std::vector<size_t> inliers;

        size_t iterations = 0;

        double inlierRatio = 0.0;

        double meanError = 0.0;
    };
    class RansacHomography
    {
    private:
        std::mt19937 m_rng;

        std::vector<geometry::Correspondence> randomSample(const std::vector<geometry::Correspondence> &correspondences,
                                                           size_t sampleSize);

        std::vector<size_t> findInliers(const std::vector<geometry::Correspondence> &correspondences,
                                        const cv::Mat &H,
                                        float threshold) const;

        bool isDegenerate(const std::vector<geometry::Correspondence> &sample);

    public:
        geometry::RansacResult estimate(const std::vector<geometry::Correspondence> &correspondences);
    };

}
