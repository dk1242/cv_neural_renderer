#include <iostream>
#include <chrono>
#include <filesystem>

#include <opencv2/opencv.hpp>

#include "vision/image_pyramid.h"
#include "vision/orb.hpp"
#include "vision/fast.hpp"
#include "vision/descriptor_matcher.h"
#include "vision/visualizer.h"

#include "geometry/ransac.hpp"
#include "geometry/homography.hpp"
#include "geometry/epipolar_geometry.hpp"
#include "geometry/fundamental_matrix.h"
#include "geometry/normalization.h"
#include "geometry/essential_matrix.hpp"
#include "geometry/pose_recovery.h"
#include "geometry/triangulation.hpp"

cv::Mat normalizeF(const cv::Mat &F)
{
    return F / cv::norm(F);
}

void testFundamentalMatrix()
{
    geometry::EpipolarGeometry epipolar;
    cv::Mat K =
        (cv::Mat_<double>(3, 3) << 800, 0, 320,
         0, 800, 240,
         0, 0, 1);

    geometry::CameraPose pose1;
    pose1.R = cv::Mat::eye(3, 3, CV_64F);
    pose1.t = cv::Mat::zeros(3, 1, CV_64F);

    double angle = 15.0 * CV_PI / 180.0;
    geometry::CameraPose pose2;
    pose2.R = (cv::Mat_<double>(3, 3) << cos(angle), 0, sin(angle),
               0, 1, 0,
               -sin(angle), 0, cos(angle));

    pose2.t = (cv::Mat_<double>(3, 1) << -1.0, 0.3, 0.2);

    std::vector<cv::Point3d> worldPoints;
    std::vector<cv::Point2d> points1;
    std::vector<cv::Point2d> points2;

    cv::RNG rng(42);

    for (int i = 0; i < 100; i++)
    {
        cv::Point3d point(
            rng.uniform(-3.0, 3.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(5.0, 15.0));
        worldPoints.push_back(point);

        points1.push_back(
            epipolar.projectPoint(
                        point,
                        pose1,
                        K)
                .imagePoint);

        points2.push_back(
            epipolar.projectPoint(
                        point,
                        pose2,
                        K)
                .imagePoint);
    }
    geometry::FundamentalMatrix fundMat;
    cv::Mat F = fundMat.estimate8Point(
        points1,
        points2);

    geometry::EssentialMatrixEstimator estimator;
    cv::Mat E = estimator.computeEssentialMatrix(F, K);

    std::cout << "essMatrix: \n"
              << E;
    cv::SVD svd(E);

    std::cout << "\nSingular values:\n"
              << svd.w << std::endl;
    cv::Mat E_corrected =
        estimator.enforceEssentialConstraints(E);

    geometry::PoseRecovery poseRecovery;

    std::vector<geometry::CameraPose> poses = poseRecovery.decomposeEssentialMatrix(E_corrected);

    geometry::Triangulator triangulator;

    // P1 = K[I|0]
    cv::Mat P1;
    cv::hconcat(
        cv::Mat::eye(3, 3, CV_64F),
        cv::Mat::zeros(3, 1, CV_64F),
        P1);

    P1 = K * P1;

    // Use Pose 2 because it matched GT
    cv::Mat P2;
    cv::hconcat(
        poses[2].R,
        poses[2].t,
        P2);

    P2 = K * P2;

    cv::Point3d X = triangulator.triangulatePoint(
        points1[0],
        points2[0],
        P1,
        P2);

    std::cout << "\n====================\n";
    std::cout << "Ground Truth Point:\n"
              << worldPoints[0]
              << "\n";

    std::cout << "Triangulated Point:\n"
              << X
              << "\n";

    std::cout << "Error: "
              << cv::norm(
                     cv::Mat(worldPoints[0]),
                     cv::Mat(X))
              << "\n";

    auto recoveredPose = poseRecovery.recoverPose(E, points1, points2, K);
    std::cout << "GT R:\n"
              << pose2.R << std::endl;
    std::cout << "Recovered R:\n"
              << recoveredPose.R << std::endl;

    std::cout << "GT t direction:\n"
              << pose2.t / cv::norm(pose2.t)
              << std::endl;

    std::cout << "Recovered t:\n"
              << recoveredPose.t
              << std::endl;
}

int main(int argc, char **argv)
{
    testFundamentalMatrix();
    return 0;
}
