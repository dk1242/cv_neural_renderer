#include "fundamental_matrix.h"

cv::Mat geometry::FundamentalMatrix::estimate8Point(const std::vector<cv::Point2d> &points1, const std::vector<cv::Point2d> &points2)
{
    Normalization norm;
    auto norm1 = norm.normalizePoints(points1);
    auto norm2 = norm.normalizePoints(points2);

    cv::Mat A = buildAMatrix(norm1.points, norm2.points);

    return A;
}

cv::Mat geometry::FundamentalMatrix::buildAMatrix(const std::vector<cv::Point2d> &points1, const std::vector<cv::Point2d> &points2)
{
    CV_Assert(points1.size() == points2.size());
    CV_Assert(points1.size() >= 8);
    cv::Mat A(points1.size(), 9, CV_64F);

    for (size_t i = 0; i < points1.size(); i++)
    {
        double u = points1[i].x;
        double v = points1[i].y;

        double up = points2[i].x;
        double vp = points2[i].y;

        A.at<double>(i, 0) = up * u;
        A.at<double>(i, 1) = up * v;
        A.at<double>(i, 2) = up;

        A.at<double>(i, 3) = vp * u;
        A.at<double>(i, 4) = vp * v;
        A.at<double>(i, 5) = vp;

        A.at<double>(i, 6) = u;
        A.at<double>(i, 7) = v;
        A.at<double>(i, 8) = 1.0;
    }
    return A;
}

cv::Mat geometry::FundamentalMatrix::enforceRank2(const cv::Mat &F)
{
    return cv::Mat();
}
