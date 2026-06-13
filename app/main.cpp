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


enum class InputMode
{
    Camera,
    ImagePair,
    SyntheticRotation
};

struct FramePair
{
    cv::Mat image1;
    cv::Mat image2;
};

void closeDisplayWindows()
{
    cv::destroyAllWindows();
    cv::waitKey(1);
}

FramePair loadImagePair()
{
    const std::filesystem::path datasetDirectory =
        std::filesystem::path(VISION_ENGINE_SOURCE_DIR) / "datasets" / "graf";

    FramePair pair;
    pair.image1 = cv::imread((datasetDirectory / "img1.ppm").string(),
                             cv::IMREAD_COLOR);
    pair.image2 = cv::imread((datasetDirectory / "img3.ppm").string(),
                             cv::IMREAD_COLOR);
    return pair;
}

FramePair createSyntheticPair(const cv::Mat &image)
{
    FramePair pair;
    pair.image1 = image.clone();
    cv::Point2f center(image.cols / 2.0f, image.rows / 2.0f);
    cv::Mat transform = cv::getRotationMatrix2D(center, 20.0, 1.0);
    cv::warpAffine(image, pair.image2, transform, image.size(), cv::INTER_NEAREST,
                   cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return pair;
}

void processPair(const cv::Mat &image1, const cv::Mat &image2)
{
    cv::Mat gray1;
    cv::Mat gray2;
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    vision::FastCornerDetector fastCornerDetector;

    vision::ImagePyramid pyramid;
    vision::OrbExtractor orbExtractor;
    vision::DescriptorMatcher matcher;
    geometry::RansacHomography ransac;

    vision::Visualizer visualizer;

    auto levels1 = pyramid.build(gray1, 8, 1.2f);
    auto levels2 = pyramid.build(gray2, 8, 1.2f);

    std::vector<vision::KeyPoint> allKeypoints1, allKeypoints2;
    std::vector<vision::OrbDescriptor> allDescriptors1, allDescriptors2;
    static const int quotas[8] = {400, 350, 300, 250, 200, 150, 100, 100};

    for (size_t level = 0; level < levels1.size(); ++level)
    {
        const auto &img1 = levels1[level].image;
        const auto &img2 = levels2[level].image;
        int quota = (level < 8) ? quotas[level] : 100;

        auto keypoints1 = fastCornerDetector.detect(img1);
        std::sort(keypoints1.begin(), keypoints1.end(), [](const auto &a, const auto &b)
                  { return a.score > b.score; });
        if (keypoints1.size() > static_cast<size_t>(quota))
        {
            keypoints1.resize(quota);
        }

        auto keypoints2 = fastCornerDetector.detect(img2);
        std::sort(keypoints2.begin(), keypoints2.end(), [](const auto &a, const auto &b)
                  { return a.score > b.score; });
        if (keypoints2.size() > static_cast<size_t>(quota))
        {
            keypoints2.resize(quota);
        }

        orbExtractor.computeOrientations(img1, keypoints1);
        orbExtractor.computeOrientations(img2, keypoints2);

        auto descriptors1 = orbExtractor.computeDescriptors(img1, keypoints1);
        auto descriptors2 = orbExtractor.computeDescriptors(img2, keypoints2);

        for (auto &kp : keypoints1)
        {
            kp.octave = static_cast<int>(level);
            kp.position.x /= levels1[level].scale;
            kp.position.y /= levels1[level].scale;
        }
        for (auto &kp : keypoints2)
        {
            kp.octave = static_cast<int>(level);
            kp.position.x /= levels2[level].scale;
            kp.position.y /= levels2[level].scale;
        }

        allKeypoints1.insert(allKeypoints1.end(), keypoints1.begin(), keypoints1.end());
        allKeypoints2.insert(allKeypoints2.end(), keypoints2.begin(), keypoints2.end());

        allDescriptors1.insert(allDescriptors1.end(), descriptors1.begin(), descriptors1.end());
        allDescriptors2.insert(allDescriptors2.end(), descriptors2.begin(), descriptors2.end());
    }

    auto matches = matcher.crosscheckMatch(allDescriptors1, allDescriptors2);
    // auto knnMatches = matcher.knnMatch(allDescriptors1, allDescriptors2, 2);
    // auto matches = matcher.ratioTest(knnMatches, 0.75f);

    std::vector<geometry::Correspondence> correspondences;
    std::vector<std::pair<int, int>> matchIndexPairs;

    correspondences.reserve(matches.size());
    matchIndexPairs.reserve(matches.size());

    for (const auto &match : matches)
    {
        correspondences.push_back(
            geometry::Correspondence{allKeypoints1[match.queryIdx].position,
                                     allKeypoints2[match.trainIdx].position});
        matchIndexPairs.emplace_back(match.queryIdx, match.trainIdx);
    }

    std::vector<int> allDist1(8, 0), allDist2(8, 0);
    for (const auto &kp : allKeypoints1)
    {
        if (kp.octave >= 0 && kp.octave < 8)
            ++allDist1[kp.octave];
    }
    for (const auto &kp : allKeypoints2)
    {
        if (kp.octave >= 0 && kp.octave < 8)
            ++allDist2[kp.octave];
    }

    std::vector<int> matchedDist1(8, 0), matchedDist2(8, 0);
    for (const auto &p : matchIndexPairs)
    {
        int o1 = allKeypoints1[p.first].octave;
        int o2 = allKeypoints2[p.second].octave;
        if (o1 >= 0 && o1 < 8)
            ++matchedDist1[o1];
        if (o2 >= 0 && o2 < 8)
            ++matchedDist2[o2];
    }

    if (correspondences.size() < 4)
        return;

    auto ransacResult = ransac.estimate(correspondences);

    // Print octave distribution of inlier keypoints (Experiment 2)
    std::vector<int> inlierDist1(8, 0), inlierDist2(8, 0);
    for (int idx : ransacResult.inliers)
    {
        if (idx >= 0 && idx < static_cast<int>(matchIndexPairs.size()))
        {
            auto p = matchIndexPairs[idx];
            int o1 = allKeypoints1[p.first].octave;
            int o2 = allKeypoints2[p.second].octave;
            if (o1 >= 0 && o1 < 8)
                ++inlierDist1[o1];
            if (o2 >= 0 && o2 < 8)
                ++inlierDist2[o2];
        }
    }

    cv::Mat image1Color;
    cv::Mat image2Color;
    cv::resize(image1, image1Color, image1.size());
    cv::resize(image2, image2Color, image2.size());

    auto visualizeMatches = matcher.visualizeMatches(image1Color, allKeypoints1,
                                                     image2Color, allKeypoints2,
                                                     matches);

    visualizer.display(visualizeMatches, "Matches", 1280, 720);
}

void processCVORBPair(const cv::Mat &image1, const cv::Mat &image2)
{
    cv::Mat gray1;
    cv::Mat gray2;
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);

    cv::Ptr<cv::ORB> orb = cv::ORB::create(1850, // nfeatures
                                           1.2f, // scaleFactor
                                           8     // nlevels
    );

    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;

    cv::Mat descriptors1;
    cv::Mat descriptors2;

    orb->detectAndCompute(
        gray1,
        cv::noArray(),
        keypoints1,
        descriptors1);
    orb->detectAndCompute(
        gray2,
        cv::noArray(),
        keypoints2,
        descriptors2);

    // keypoints per octave
    std::vector<int> octaveDist1(8, 0);
    std::vector<int> octaveDist2(8, 0);
    for (const auto &kp : keypoints1)
    {
        if (kp.octave >= 0 && kp.octave < 8)
            ++octaveDist1[kp.octave];
    }
    for (const auto &kp : keypoints2)
    {
        if (kp.octave >= 0 && kp.octave < 8)
            ++octaveDist2[kp.octave];
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    std::vector<cv::DMatch> goodMatches;
    for (const auto &knnMatch : knnMatches)
    {
        if (knnMatch.size() == 2 && knnMatch[0].distance < 0.75f * knnMatch[1].distance)
        {
            goodMatches.push_back(knnMatch[0]);
        }
    }

    cv::Mat visualizeMatches;
    cv::drawMatches(image1, keypoints1, image2, keypoints2,
                    goodMatches, visualizeMatches);

    cv::namedWindow("CV ORB Matches", cv::WINDOW_NORMAL);
    cv::resizeWindow("CV ORB Matches", 1280, 720);
    cv::imshow("CV ORB Matches", visualizeMatches);
}

cv::Point toImagePoint(const cv::Point2d &p)
{
    constexpr double scale = 200.0;

    int x = static_cast<int>(400 + p.x * scale);
    int y = static_cast<int>(300 - p.y * scale);

    return {x, y};
}

cv::Point2i toScreen(double x, double z)
{
    constexpr double scale = 100.0;

    int sx = 600 + static_cast<int>(x * scale);
    int sy = 700 - static_cast<int>(z * scale);

    return {sx, sy};
}

void visualizeEpipolarGeometry()
{
    cv::namedWindow("Top View", cv::WINDOW_NORMAL);
    cv::resizeWindow("Top View", 800, 1200);

    cv::namedWindow("Image 1", cv::WINDOW_NORMAL);
    cv::resizeWindow("Image 1", 600, 800);

    cv::namedWindow("Image 2", cv::WINDOW_NORMAL);
    cv::resizeWindow("Image 2", 600, 800);

    cv::Mat canvas(800, 1200, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat img1(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat img2(600, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    geometry::EpipolarGeometry epipolar;

    geometry::CameraPose cam1;
    cam1.R = cv::Mat::eye(3, 3, CV_64F);
    cam1.t = cv::Mat::zeros(3, 1, CV_64F);

    geometry::CameraPose cam2;
    double angle = 10.0 * CV_PI / 180.0;

    cam2.R = (cv::Mat_<double>(3, 3) << cos(angle), 0, sin(angle),
              0, 1, 0,
              -sin(angle), 0, cos(angle));
    cam2.t = (cv::Mat_<double>(3, 1) << -1.0, 0.0, 0.0);

    cv::Point3d C1 = epipolar.cameraCenter(cam1);
    cv::Point3d C2 = epipolar.cameraCenter(cam2);
    std::cout
        << "C1 = "
        << C1.x << ", "
        << C1.y << ", "
        << C1.z << '\n';
    std::cout
        << "C2 = "
        << C2.x << ", "
        << C2.y << ", "
        << C2.z << '\n';
    double c1x = C1.x;
    double c1z = C1.z;

    double c2x = C2.x;
    double c2z = C2.z;

    cv::circle(canvas, toScreen(c1x, c1z), 8, {0, 0, 255}, -1);
    cv::circle(canvas, toScreen(c2x, c2z), 8, {255, 0, 0}, -1);

    cv::line(canvas, toScreen(c1x, c1z), toScreen(c2x, c2z), {0, 0, 0}, 2);

    std::vector<cv::Point3d> points = {{0.0, 0.0, 4.0}, {1.0, 0.0, 5.0}, {-1.0, 0.0, 6.0}, {2.0, 0.0, 8.0}};

    for (const auto &p : points)
    {
        cv::circle(canvas, toScreen(p.x, p.z), 5, {0, 180, 0}, -1);
        cv::line(canvas, toScreen(c1x, c1z), toScreen(p.x, p.z), {255, 0, 255}, 1);
        cv::line(canvas, toScreen(c2x, c2z), toScreen(p.x, p.z), {255, 255, 0}, 1);
    }

    cv::Mat K = (cv::Mat_<double>(3, 3) << 800, 0, 400,
                 0, 800, 300,
                 0, 0, 1);
    // cv::Point3d P(3.0, 1.5, 5.0);
    std::vector<cv::Point3d> worldPoints =
        {
            {-2.0, -0.5, 4.0},
            {-1.0, 0.2, 5.0},
            {0.0, -0.3, 6.0},
            {1.0, 0.4, 7.0},
            {2.0, -0.1, 8.0}};

    std::vector<cv::Scalar> colors =
        {
            {255, 0, 0},
            {0, 255, 0},
            {0, 0, 255},
            {255, 0, 255},
            {0, 255, 255}};

    for (size_t pointIndex = 0;
         pointIndex < worldPoints.size();
         ++pointIndex)
    {
        const auto &P = worldPoints[pointIndex];
        const auto &color = colors[pointIndex];

        auto x1 =
            epipolar.projectPoint(P, cam1, K);

        auto x2 =
            epipolar.projectPoint(P, cam2, K);

        if (!x1.visible || !x2.visible)
        {
            continue;
        }

        std::cout
            << "\n====================\n"
            << "Point " << pointIndex
            << " : "
            << P.x << ", "
            << P.y << ", "
            << P.z << '\n';

        cv::circle(
            img1,
            cv::Point(
                static_cast<int>(x1.imagePoint.x),
                static_cast<int>(x1.imagePoint.y)),
            5,
            color,
            -1);

        cv::circle(
            img2,
            cv::Point(
                static_cast<int>(x2.imagePoint.x),
                static_cast<int>(x2.imagePoint.y)),
            5,
            color,
            -1);

        cv::Vec3d rayDirection(
            P.x - C1.x,
            P.y - C1.y,
            P.z - C1.z);

        rayDirection /= cv::norm(rayDirection);

        std::vector<cv::Point2d> projectedRayPoints;

        std::vector<double> depths =
            {
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
                20.0};

        for (double d : depths)
        {
            cv::Point3d X(
                C1.x + d * rayDirection[0],
                C1.y + d * rayDirection[1],
                C1.z + d * rayDirection[2]);

            auto p2 =
                epipolar.projectPoint(X, cam2, K);

            if (!p2.visible)
            {
                continue;
            }

            projectedRayPoints.push_back(
                p2.imagePoint);

            std::cout
                << "depth="
                << d
                << " -> "
                << p2.imagePoint.x
                << ", "
                << p2.imagePoint.y
                << '\n';

            cv::circle(
                img2,
                cv::Point(
                    static_cast<int>(p2.imagePoint.x),
                    static_cast<int>(p2.imagePoint.y)),
                3,
                color,
                -1);
        }

        for (size_t i = 1;
             i < projectedRayPoints.size();
             ++i)
        {
            cv::line(
                img2,
                cv::Point(
                    static_cast<int>(
                        projectedRayPoints[i - 1].x),
                    static_cast<int>(
                        projectedRayPoints[i - 1].y)),
                cv::Point(
                    static_cast<int>(
                        projectedRayPoints[i].x),
                    static_cast<int>(
                        projectedRayPoints[i].y)),
                color,
                2);
        }
    }
    cv::Mat baseline =
        (cv::Mat_<double>(3, 1)
             << C2.x - C1.x,
         C2.y - C1.y,
         C2.z - C1.z);

    std::cout << "\nBaseline:\n"
              << baseline
              << "\n";
    std::cout
        << "\nCamera 2 Rotation:\n"
        << cam2.R
        << "\n";
    cv::Point2d e = epipolar.computeEpipole(cam1, cam2, K);
    cv::circle(img2, cv::Point(static_cast<int>(e.x), static_cast<int>(e.y)), 8, {255, 0, 0}, -1);

    cv::imshow("Top View", canvas);
    cv::imshow("Image 1", img1);
    cv::imshow("Image 2", img2);

    cv::waitKey(0);
}

int main(int argc, char **argv)
{
    InputMode mode = InputMode::Camera;

    if (argc > 1)
    {
        std::string arg = argv[1];

        std::cout << "arg  " << arg;

        if (arg == "--images")
        {
            mode = InputMode::ImagePair;
        }
        else if (arg == "--synthetic")
        {
            mode = InputMode::SyntheticRotation;
        }
    }

    if (mode == InputMode::ImagePair)
    {
        auto pair = loadImagePair();

        if (pair.image1.empty() ||
            pair.image2.empty())
        {
            std::cout
                << "Failed to load images\n";
            return -1;
        }

        processPair(pair.image1, pair.image2);

        cv::waitKey(0);
        closeDisplayWindows();

        return 0;
    }
    else
    {
        std::cout << "Herre....\n";
        visualizeEpipolarGeometry();
        return 0;
    }

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cout
            << "Cannot open camera\n";
        return -1;
    }

    cv::Mat frame;
    cv::Mat previousFrame;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
        {
            break;
        }

        if (mode ==
            InputMode::SyntheticRotation)
        {
            auto pair =
                createSyntheticPair(frame);

            processPair(pair.image1,
                        pair.image2);
        }
        else
        {
            if (previousFrame.empty())
            {
                previousFrame =
                    frame.clone();

                continue;
            }

            processPair(previousFrame,
                        frame);

            previousFrame =
                frame.clone();
        }

        int key =
            cv::waitKey(1);

        if (key == 27)
        {
            break;
        }
    }

    cap.release();

    closeDisplayWindows();

    return 0;
}
