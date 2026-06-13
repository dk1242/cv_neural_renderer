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

    std::vector<cv::Point2d> points1;
    std::vector<cv::Point2d> points2;

    cv::RNG rng(42);

    for (int i = 0; i < 100; i++)
    {
        cv::Point3d point(
            rng.uniform(-3.0, 3.0),
            rng.uniform(-2.0, 2.0),
            rng.uniform(5.0, 15.0));

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

    std::cout
        << "Estimated F:\n"
        << F
        << '\n';
    cv::Mat Festimated = F;
    cv::Mat Fgt = fundMat.computeFundamentalGroundTruth(K, pose2.R, pose2.t);

    cv::Mat FestNorm = normalizeF(Festimated);
    cv::Mat FgtNorm = normalizeF(Fgt);
    std::cout
        << "Festimated F:\n"
        << Festimated
        << '\n';
    std::cout
        << "Fgt F:\n"
        << Fgt
        << '\n';
    double sameSign =
        cv::norm(FestNorm - FgtNorm);

    double oppositeSign =
        cv::norm(FestNorm + FgtNorm);

    double error =
        std::min(sameSign, oppositeSign);

    std::cout
        << "same sign: "
        << sameSign
        << "\nopp Sign "
        << oppositeSign
        << "\nF comparison error: "
        << error
        << std::endl;

    cv::SVD svdEst(Festimated);
    cv::SVD svdGt(Fgt);

    std::cout << svdEst.w << std::endl;
    std::cout << svdGt.w << std::endl;
}

int main(int argc, char **argv)
{
    testFundamentalMatrix();
    return 0;
}
