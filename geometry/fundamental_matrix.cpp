#include "fundamental_matrix.h"

cv::Mat skewSymmetric(const cv::Mat &t)
{
    double tx = t.at<double>(0);
    double ty = t.at<double>(1);
    double tz = t.at<double>(2);

    return (cv::Mat_<double>(3, 3) << 0, -tz, ty,
            tz, 0, -tx,
            -ty, tx, 0);
}
cv::Mat computeEssentialGroundTruth(
    const cv::Mat &R,
    const cv::Mat &t)
{
    return skewSymmetric(t) * R;
}

cv::Mat geometry::FundamentalMatrix::estimate8Point(const std::vector<cv::Point2d> &points1, const std::vector<cv::Point2d> &points2)
{
    Normalization norm;
    auto norm1 = norm.normalizePoints(points1);
    auto norm2 = norm.normalizePoints(points2);

    cv::Mat A = buildAMatrix(norm1.points, norm2.points);
    // std::cout
    //     << "\nA Matrix:\n"
    //     << A
    //     << '\n';
    cv::SVD svd(A, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // std::cout << "Singular values:\n";
    // std::cout << svd.w << '\n';

    cv::Mat f = svd.vt.row(8).t();

    cv::Mat F(3, 3, CV_64F);

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            F.at<double>(r, c) =
                f.at<double>(3 * r + c);
        }
    }

    F = enforceRank2(F);

    F = norm2.T.t() *
        F *
        norm1.T;

    F /= cv::norm(F);

    return F;
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
    cv::SVD svd(F);

    cv::Mat W = cv::Mat::diag(svd.w);

    W.at<double>(2, 2) = 0.0;

    cv::Mat F_rank2 =
        svd.u *
        W *
        svd.vt;

    return F_rank2;
}

cv::Mat geometry::FundamentalMatrix::computeFundamentalGroundTruth(
    const cv::Mat &K,
    const cv::Mat &R,
    const cv::Mat &t)
{
    cv::Mat E = computeEssentialGroundTruth(R, t);

    cv::Mat Kinv = K.inv();

    return Kinv.t() * E * Kinv;
}