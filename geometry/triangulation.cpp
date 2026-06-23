#include "triangulation.hpp"

cv::Point3d geometry::Triangulator::triangulatePoint(const cv::Point2d &x1, const cv::Point2d &x2,
                                                     const cv::Mat &P1, const cv::Mat &P2)
{
    cv::Mat A(4, 4, CV_64F);
    A.row(0) = x1.x * P1.row(2) - P1.row(0);
    A.row(1) = x1.y * P1.row(2) - P1.row(1);
    A.row(2) = x2.x * P2.row(2) - P2.row(0);
    A.row(3) = x2.y * P2.row(2) - P2.row(1);

    cv::SVD svd(A);

    cv::Mat X_h = svd.vt.row(3).t();

    X_h /= X_h.at<double>(3);

    return cv::Point3d(
        X_h.at<double>(0),
        X_h.at<double>(1),
        X_h.at<double>(2));
}
