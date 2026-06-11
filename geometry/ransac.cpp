#include "ransac.hpp"
#include <numeric>

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
    geometry::HomographyEstimator estimator;
    for (size_t iter = 0; iter < 1000; ++iter)
    {
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

        if (inliers.size() > bestResult.inliers.size())
        {
            bestResult.inliers = inliers;
            bestResult.homography = H;
            bestResult.meanError = meanError;
        }
        else if (inliers.size() == bestResult.inliers.size())
        {
            if (meanError < bestResult.meanError)
            {
                bestResult.inliers = inliers;
                bestResult.homography = H;
                bestResult.meanError = meanError;
            }
        }
    }
    std::vector<geometry::Correspondence> inlierCorrespondences;
    for (size_t idx : bestResult.inliers)
    {
        inlierCorrespondences.push_back(correspondences[idx]);
    }
    if (inlierCorrespondences.size() >= 4 && !correspondences.empty())
    {
        bestResult.homography = estimator.estimate(inlierCorrespondences);
        bestResult.inliers = findInliers(correspondences, bestResult.homography, 3.0f);
        bestResult.meanError = computeMeanError(correspondences, bestResult.inliers,
                                                bestResult.homography);
        bestResult.inlierRatio = static_cast<float>(bestResult.inliers.size()) / correspondences.size();
    }
    return bestResult;
}
