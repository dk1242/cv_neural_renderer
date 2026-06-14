#include "essential_matrix.hpp"

cv::Mat geometry::EssentialMatrixEstimator::computeEssentialMatrix(const cv::Mat &F,
                                                                   const cv::Mat &K)
{
    cv::Mat K_t = K.t();
    cv::Mat E = K_t * F * K;
    return E;
}

cv::Mat geometry::EssentialMatrixEstimator::enforceEssentialConstraints(const cv::Mat &E)
{
    cv::SVD svd(E);

    cv::Mat U = svd.u.clone();
    cv::Mat Vt = svd.vt.clone();

    if (cv::determinant(U) < 0)
    {
        U.col(2) *= -1;
    }

    if (cv::determinant(Vt) < 0)
    {
        Vt.row(2) *= -1;
    }

    double s =
        (svd.w.at<double>(0) +
         svd.w.at<double>(1)) *
        0.5;

    cv::Mat Sigma = cv::Mat::zeros(3, 3, CV_64F);

    Sigma.at<double>(0, 0) = s;
    Sigma.at<double>(1, 1) = s;
    Sigma.at<double>(2, 2) = 0.0;

    return U * Sigma * Vt;
}