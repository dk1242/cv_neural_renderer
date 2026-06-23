#pragma once

#include <opencv2/opencv.hpp>
#include <limits>
#include "correspondence.hpp"
#include "homography.hpp"
#include "fundamental_matrix.h"
#include <random>

namespace geometry
{
    struct RansacResult
    {
        cv::Mat model;
        cv::Mat inlierMask;
        size_t iterations = 0;
        size_t inlierCount = 0;
        double inlierRatio = 0.0;
        double meanError = std::numeric_limits<double>::max();
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
        double computeMeanError(const std::vector<geometry::Correspondence> &correspondences,
                                const std::vector<size_t> &inliers, const cv::Mat &H);

    public:
        geometry::RansacResult estimate(const std::vector<geometry::Correspondence> &correspondences);
    };

    class RansacFundamental
    {
    private:
        std::mt19937 m_rng{std::random_device{}()};

    public:
        RansacResult estimateFundamental(
            const std::vector<cv::Point2d> &points1,
            const std::vector<cv::Point2d> &points2,
            double threshold = 3.0,
            size_t maxIterations = 2000);
    };

}
