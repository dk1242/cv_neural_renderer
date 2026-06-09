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
        if (error < threshold)
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

geometry::RansacResult geometry::RansacHomography::estimate(const std::vector<geometry::Correspondence> &correspondences)
{
    geometry::RansacResult bestResult;
    for (size_t iter = 0; iter < 1000; ++iter)
    {
        // sample 4 points
        auto sample = randomSample(correspondences, 4);

        // reject degenerate
        if (isDegenerate(sample))
        {
            return bestResult;
        }
        // estimate H
        cv::Mat H = geometry::HomographyEstimator().estimate(sample);
        // find inliers
        auto inliers = findInliers(correspondences, H, 3.0f);
        // if better than best,
        if (inliers.size() > bestResult.inliers.size())
        {
            bestResult.inliers = inliers;
            bestResult.homography = H;
        }
    }

    // update best model
    return bestResult;
}
