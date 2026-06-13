#include "normalization.h"

geometry::NormalizationResult geometry::Normalization::normalizePoints(const std::vector<cv::Point2d> &points)
{
    double meanX = 0.0;
    double meanY = 0.0;

    for (const auto &p : points)
    {
        meanX += p.x;
        meanY += p.y;
    }

    meanX /= points.size();
    meanY /= points.size();

    double meanDistance = 0.0;

    for (const auto &p : points)
    {
        double dx = p.x - meanX;
        double dy = p.y - meanY;

        meanDistance += std::sqrt(dx * dx + dy * dy);
    }
    if (meanDistance < 1e-12)
    {
        throw std::runtime_error(
            "Degenerate point configuration");
    }
    meanDistance /= points.size();
    double scale = std::sqrt(2.0) / meanDistance;
    cv::Mat T = (cv::Mat_<double>(3, 3) << scale, 0.0, -scale * meanX,
                 0.0, scale, -scale * meanY,
                 0.0, 0.0, 1.0);

    std::vector<cv::Point2d> normalizedPoints;
    normalizedPoints.reserve(points.size());
    for (const auto &p : points)
    {
        cv::Mat point = (cv::Mat_<double>(3, 1) << p.x, p.y, 1.0);
        cv::Mat pNorm = T * point;
        normalizedPoints.emplace_back(
            pNorm.at<double>(0),
            pNorm.at<double>(1));
    }

    NormalizationResult result;
    result.points = std::move(normalizedPoints);
    result.T = T;

    return result;
}