#include "homography.hpp"

cv::Point2f geometry::HomographyEstimator::projectPoint(const cv::Point2f &point, const cv::Mat &H) const
{
    double x = point.x;
    double y = point.y;

    double X = H.at<double>(0, 0) * x +
               H.at<double>(0, 1) * y +
               H.at<double>(0, 2);

    double Y = H.at<double>(1, 0) * x +
               H.at<double>(1, 1) * y +
               H.at<double>(1, 2);

    double W = H.at<double>(2, 0) * x +
               H.at<double>(2, 1) * y +
               H.at<double>(2, 2);

    constexpr double epsilon = 1e-8;

    if (std::abs(W) < epsilon)
    {
        return {};
    }
    return cv::Point2f(static_cast<float>(X / W),
                       static_cast<float>(Y / W));
}

float geometry::HomographyEstimator::reprojectionError(const geometry::Correspondence &correspondence, const cv::Mat &H) const
{
    cv::Point2f predicted = projectPoint(correspondence.source, H);
    float errorX = predicted.x - correspondence.destination.x;
    float errorY = predicted.y - correspondence.destination.y;
    return (errorX * errorX + errorY * errorY);
}

cv::Mat geometry::HomographyEstimator::estimate(
    const std::vector<geometry::Correspondence> &correspondences)
{
    if (correspondences.size() < 4)
    {
        throw std::runtime_error("At least 4 correspondences are required to estimate homography.");
    }
    cv::Mat A(correspondences.size() * 2, 9, CV_64F);
    for (size_t i = 0; i < correspondences.size(); ++i)
    {
        const auto &corr = correspondences[i];
        double x = corr.source.x;
        double y = corr.source.y;
        double u = corr.destination.x;
        double v = corr.destination.y;

        A.at<double>(2 * i, 0) = x;
        A.at<double>(2 * i, 1) = y;
        A.at<double>(2 * i, 2) = 1;
        A.at<double>(2 * i, 3) = 0;
        A.at<double>(2 * i, 4) = 0;
        A.at<double>(2 * i, 5) = 0;
        A.at<double>(2 * i, 6) = -u * x;
        A.at<double>(2 * i, 7) = -u * y;
        A.at<double>(2 * i, 8) = -u;

        A.at<double>(2 * i + 1, 0) = 0;
        A.at<double>(2 * i + 1, 1) = 0;
        A.at<double>(2 * i + 1, 2) = 0;
        A.at<double>(2 * i + 1, 3) = x;
        A.at<double>(2 * i + 1, 4) = y;
        A.at<double>(2 * i + 1, 5) = 1;
        A.at<double>(2 * i + 1, 6) = -v * x;
        A.at<double>(2 * i + 1, 7) = -v * y;
        A.at<double>(2 * i + 1, 8) = -v;
    }
    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat h = svd.vt.row(8).t(); // Last row of V^T (or last column of V)
    cv::Mat H = h.reshape(0, 3);

    double scale = H.at<double>(2, 2);

    if (std::abs(scale) > 1e-8)
    {
        H /= scale;
    }
    return H;
}