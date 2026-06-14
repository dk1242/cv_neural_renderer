#include "pose_recovery.h"

int geometry::PoseRecovery::countPositiveDepth(const geometry::CameraPose &pose, const std::vector<cv::Point2d> &points1,
                                               const std::vector<cv::Point2d> &points2, const cv::Mat &K)
{
    cv::Mat P1 = cv::Mat::zeros(3, 4, CV_64F);
    cv::Mat I = cv::Mat::eye(3, 3, CV_64F);
    I.copyTo(P1.colRange(0, 3));

    cv::Mat P2 = cv::Mat::zeros(3, 4, CV_64F);
    pose.R.copyTo(P2.colRange(0, 3));
    pose.t.copyTo(P2.col(3));

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    int count = 0;

    for (size_t i = 0; i < points1.size(); i++)
    {
        cv::Point2d p1(
            (points1[i].x - cx) / fx,
            (points1[i].y - cy) / fy);

        cv::Point2d p2(
            (points2[i].x - cx) / fx,
            (points2[i].y - cy) / fy);

        cv::Vec3d X = m_triangulator.triangulatePoint(p1, p2, P1, P2);

        double z1 = X[2];

        cv::Mat Xmat =
            (cv::Mat_<double>(3, 1)
                 << X[0],
             X[1], X[2]);

        cv::Mat Xc2 =
            pose.R * Xmat + pose.t;

        double z2 = Xc2.at<double>(2);

        if (z1 > 0.0 && z2 > 0.0)
        {
            count++;
        }
    }

    return count;
}

std::vector<geometry::CameraPose> geometry::PoseRecovery::decomposeEssentialMatrix(
    const cv::Mat &E)
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
    cv::Mat W = (cv::Mat_<double>(3, 3) << 0, -1, 0,
                 1, 0, 0,
                 0, 0, 1);
    cv::Mat R1 = U * W * Vt;
    cv::Mat R2 = U * W.t() * Vt;
    if (cv::determinant(R1) < 0)
    {
        R1 *= -1;
    }

    if (cv::determinant(R2) < 0)
    {
        R2 *= -1;
    }
    cv::Mat t = U.col(2).clone();
    return {
        {R1, t},
        {R1, -t},
        {R2, t},
        {R2, -t}};
}

geometry::CameraPose geometry::PoseRecovery::recoverPose(const cv::Mat &E, const std::vector<cv::Point2d> &points1,
                                                         const std::vector<cv::Point2d> &points2, const cv::Mat &K)
{
    auto candidates = decomposeEssentialMatrix(E);
    geometry::CameraPose bestPose;
    int bestCount = -1;

    for (size_t i = 0; i < candidates.size(); i++)
    {
        int count = countPositiveDepth(candidates[i], points1, points2, K);
        std::cout
            << "Candidate "
            << i
            << " : "
            << count
            << std::endl;
        if (count > bestCount)
        {
            bestCount = count;
            bestPose = candidates[i];
        }
    }
    // std::cout << "bestCount " << bestCount;
    return bestPose;
}
