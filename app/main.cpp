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

int main(int argc, char **argv)
{
    InputMode mode =
        InputMode::Camera;

    if (argc > 1)
    {
        std::string arg = argv[1];

        if (arg == "--images")
        {
            mode =
                InputMode::ImagePair;
        }
        else if (arg == "--synthetic")
        {
            mode =
                InputMode::SyntheticRotation;
        }
    }

    if (mode ==
        InputMode::ImagePair)
    {
        auto pair =
            loadImagePair();

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
