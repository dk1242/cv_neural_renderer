#include "ransac.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace
{
double sampsonError(const cv::Mat &F, const cv::Point2d &point1, const cv::Point2d &point2)
{
    const cv::Matx33d matrix(
        F.at<double>(0, 0), F.at<double>(0, 1), F.at<double>(0, 2),
        F.at<double>(1, 0), F.at<double>(1, 1), F.at<double>(1, 2),
        F.at<double>(2, 0), F.at<double>(2, 1), F.at<double>(2, 2));

    const cv::Vec3d x1(point1.x, point1.y, 1.0);
    const cv::Vec3d x2(point2.x, point2.y, 1.0);
    const cv::Vec3d Fx1 = matrix * x1;
    const cv::Vec3d Ftx2 = matrix.t() * x2;
    const double residual = x2.dot(Fx1);
    const double denominator =
        Fx1[0] * Fx1[0] + Fx1[1] * Fx1[1] +
        Ftx2[0] * Ftx2[0] + Ftx2[1] * Ftx2[1];

    if (denominator <= std::numeric_limits<double>::epsilon())
    {
        return std::numeric_limits<double>::max();
    }

    return (residual * residual) / denominator;
}

std::vector<size_t> findFundamentalInliers(
    const std::vector<cv::Point2d> &points1,
    const std::vector<cv::Point2d> &points2,
    const cv::Mat &F,
    double threshold,
    double &meanError)
{
    std::vector<size_t> inliers;
    double totalError = 0.0;
    const double squaredThreshold = threshold * threshold;

    for (size_t i = 0; i < points1.size(); ++i)
    {
        const double error = sampsonError(F, points1[i], points2[i]);
        if (error <= squaredThreshold)
        {
            inliers.push_back(i);
            totalError += std::sqrt(error);
        }
    }

    meanError = inliers.empty()
                    ? std::numeric_limits<double>::max()
                    : totalError / static_cast<double>(inliers.size());
    return inliers;
}
}

std::vector<geometry::Correspondence> geometry::RansacHomography::randomSample(
    const std::vector<geometry::Correspondence> &correspondences, size_t sampleSize)
{
    std::vector<size_t> indices(correspondences.size());

    std::iota(indices.begin(), indices.end(), 0);

    std::shuffle(indices.begin(), indices.end(), m_rng);

    std::vector<geometry::Correspondence> sample;
    for (size_t i = 0; i < sampleSize; ++i)
    {
        sample.push_back(correspondences[indices[i]]);
    }
    return sample;
}

std::vector<size_t> geometry::RansacHomography::findInliers(const std::vector<geometry::Correspondence> &correspondences, const cv::Mat &H, float threshold) const
{
    std::vector<size_t> inliers;
    geometry::HomographyEstimator estimator;
    for (size_t i = 0; i < correspondences.size(); ++i)
    {
        const geometry::Correspondence &corr = correspondences[i];
        float error = estimator.reprojectionError(corr, H);
        if (error < threshold * threshold)
        {
            inliers.push_back(i);
        }
    }
    return inliers;
}

bool geometry::RansacHomography::isDegenerate(const std::vector<geometry::Correspondence> &sample)
{
    cv::Point2f p1, p2, p3;
    p1 = sample[0].source;
    p2 = sample[1].source;
    p3 = sample[2].source;

    float area = std::abs((p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x));
    return area < 1e-8; // Check if the area is negligible (degenerate case)
}

double geometry::RansacHomography::computeMeanError(const std::vector<geometry::Correspondence> &correspondences,
                                                    const std::vector<size_t> &inliers, const cv::Mat &H)
{
    if (inliers.empty())
    {
        return std::numeric_limits<double>::max();
    }
    geometry::HomographyEstimator estimator;
    double totalError = 0.0;
    for (size_t idx : inliers)
    {
        totalError += sqrt(estimator.reprojectionError(correspondences[idx], H));
    }
    return totalError / inliers.size();
}

geometry::RansacResult geometry::RansacHomography::estimate(const std::vector<geometry::Correspondence> &correspondences)
{
    geometry::RansacResult bestResult;
    if (correspondences.size() < 4)
    {
        return bestResult;
    }

    geometry::HomographyEstimator estimator;
    std::vector<size_t> bestInliers;

    for (size_t iter = 0; iter < 1000; ++iter)
    {
        bestResult.iterations = iter + 1;

        // sample 4 points
        auto sample = randomSample(correspondences, 4);

        // reject degenerate
        if (isDegenerate(sample))
        {
            continue;
        }
        // estimate H
        cv::Mat H = estimator.estimate(sample);
        // find inliers
        auto inliers = findInliers(correspondences, H, 3.0f);
        double meanError = computeMeanError(correspondences, inliers, H);

        if (inliers.size() > bestInliers.size())
        {
            bestInliers = std::move(inliers);
            bestResult.model = H;
            bestResult.meanError = meanError;
        }
        else if (inliers.size() == bestInliers.size())
        {
            if (meanError < bestResult.meanError)
            {
                bestInliers = std::move(inliers);
                bestResult.model = H;
                bestResult.meanError = meanError;
            }
        }
    }

    std::vector<geometry::Correspondence> inlierCorrespondences;
    for (size_t idx : bestInliers)
    {
        inlierCorrespondences.push_back(correspondences[idx]);
    }

    if (inlierCorrespondences.size() >= 4 && !correspondences.empty())
    {
        bestResult.model = estimator.estimate(inlierCorrespondences);
        bestInliers = findInliers(correspondences, bestResult.model, 3.0f);
        bestResult.meanError = computeMeanError(
            correspondences, bestInliers, bestResult.model);
        bestResult.inlierCount = bestInliers.size();
        bestResult.inlierRatio =
            static_cast<double>(bestResult.inlierCount) /
            static_cast<double>(correspondences.size());

        bestResult.inlierMask =
            cv::Mat::zeros(static_cast<int>(correspondences.size()), 1, CV_8U);
        for (const size_t index : bestInliers)
        {
            bestResult.inlierMask.at<uchar>(static_cast<int>(index)) = 1;
        }
    }

    return bestResult;
}

geometry::RansacResult geometry::RansacFundamental::estimateFundamental(
    const std::vector<cv::Point2d> &points1,
    const std::vector<cv::Point2d> &points2,
    double threshold,
    size_t maxIterations)
{
    RansacResult result;
    if (points1.size() != points2.size() || points1.size() < 8)
    {
        return result;
    }

    std::vector<size_t> indices(points1.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<size_t> bestInliers;

    for (size_t iteration = 0; iteration < maxIterations; ++iteration)
    {
        result.iterations = iteration + 1;
        std::shuffle(indices.begin(), indices.end(), m_rng);

        std::vector<cv::Point2d> sample1;
        std::vector<cv::Point2d> sample2;
        sample1.reserve(8);
        sample2.reserve(8);
        for (size_t i = 0; i < 8; ++i)
        {
            sample1.push_back(points1[indices[i]]);
            sample2.push_back(points2[indices[i]]);
        }

        cv::Mat candidate;
        try
        {
            candidate = FundamentalMatrix::estimate8Point(sample1, sample2);
        }
        catch (const cv::Exception &)
        {
            continue;
        }

        if (candidate.empty() || !cv::checkRange(candidate))
        {
            continue;
        }

        double meanError = 0.0;
        auto inliers = findFundamentalInliers(
            points1, points2, candidate, threshold, meanError);

        if (inliers.size() > bestInliers.size() ||
            (inliers.size() == bestInliers.size() && meanError < result.meanError))
        {
            bestInliers = std::move(inliers);
            result.model = candidate;
            result.meanError = meanError;
        }
    }

    if (bestInliers.size() < 8)
    {
        result.model.release();
        return result;
    }

    std::vector<cv::Point2d> inlierPoints1;
    std::vector<cv::Point2d> inlierPoints2;
    inlierPoints1.reserve(bestInliers.size());
    inlierPoints2.reserve(bestInliers.size());
    for (const size_t index : bestInliers)
    {
        inlierPoints1.push_back(points1[index]);
        inlierPoints2.push_back(points2[index]);
    }

    result.model = FundamentalMatrix::estimate8Point(inlierPoints1, inlierPoints2);
    double refinedMeanError = 0.0;
    bestInliers = findFundamentalInliers(
        points1, points2, result.model, threshold, refinedMeanError);
    result.meanError = refinedMeanError;

    result.inlierMask = cv::Mat::zeros(static_cast<int>(points1.size()), 1, CV_8U);
    for (const size_t index : bestInliers)
    {
        result.inlierMask.at<uchar>(static_cast<int>(index)) = 1;
    }
    result.inlierCount = bestInliers.size();
    result.inlierRatio =
        static_cast<double>(result.inlierCount) / static_cast<double>(points1.size());

    return result;
}
