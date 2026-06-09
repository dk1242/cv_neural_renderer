#include "vision/gaussian.hpp"
#include "vision/sobel.hpp"
#include "vision/visualizer.h"
#include "vision/canny.hpp"
#include "vision/harris.hpp"
#include "vision/fast.hpp"
#include "vision/image_pyramid.h"
#include "vision/orb.hpp"
#include "vision/descriptor_matcher.h"
#include "geometry/ransac.hpp"
#include "geometry/homography.hpp"

using namespace vision;

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cout << "Cannot open camera\n";
        return -1;
    }

    // cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    // cv::resizeWindow("Camera", 640, 720);

    cv::Mat frame, gray, blurred, edges, display;

    vision::ImagePyramid pyramid;
    vision::OrbExtractor orbExtractor;
    vision::DescriptorMatcher descriptorMatcher;

    geometry::RansacHomography ransacHomography;
    while (true)
    {
        cap >> frame;

        if (frame.empty())
        {
            break;
        }

        if (frame.channels() == 1)
        {
            gray = frame;
        }
        else
        {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        }

        // CannyResult result = vision::CannyDetector().detectDebug(gray, 10.0f, 100.0f);
        // vision::HarrisCornerDetector harrisDetector;
        // cv::Mat harrisResponse = harrisDetector.computeResponse(gray);
        // std::vector<cv::Point> corners = harrisDetector.detectCorners(harrisResponse, 0.01f);
        // cv::Mat visualizedHarrisCorners = harrisDetector.visualizeCorners(frame, corners);

        auto levels = pyramid.build(gray, 1, 1.2f);
        for (int i = 0; i < levels.size(); ++i)
        {
            const auto &level = levels[i];
            cv::Mat shiftedGray;
            cv::Mat shiftedFrame;
            cv::Point2f center(level.image.cols / 2.0f,
                               level.image.rows / 2.0f);

            cv::Mat transform = cv::getRotationMatrix2D(center,
                                                        20.0, // degrees
                                                        1.0); // scale
            cv::warpAffine(level.image, shiftedGray, transform, level.image.size(),
                           cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
            cv::warpAffine(frame, shiftedFrame, transform, frame.size(),
                           cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

            std::vector<vision::KeyPoint> corners = vision::FastCornerDetector().detect(level.image);
            for (auto &kp : corners)
                kp.octave = i;
            // cv::Mat visualizedCorners = vision::FastCornerDetector().visualizeCorners(level.image, corners);
            orbExtractor.computeOrientations(level.image, corners);
            cv::Mat visualizedCorners = orbExtractor.visualizeArrows(frame, corners);
            cv::Mat visualizeDescriptors = orbExtractor.visualizeDescriptorPattern(frame, corners);

            auto corners2 = vision::FastCornerDetector().detect(shiftedGray);
            for (auto &kp : corners2)
                kp.octave = i;
            orbExtractor.computeOrientations(shiftedGray, corners2);
            std::vector<vision::OrbDescriptor> descriptors1 = orbExtractor.computeDescriptors(level.image, corners);
            std::vector<vision::OrbDescriptor> descriptors2 = orbExtractor.computeDescriptors(shiftedGray, corners2);

            // auto knnMatches = descriptorMatcher.knnMatch(descriptors1, descriptors2, 2);
            // auto ratioMatches = descriptorMatcher.ratioTest(knnMatches, 0.75f);
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<vision::Match> matches = descriptorMatcher.crosscheckMatch(descriptors1, descriptors2);
            auto end = std::chrono::high_resolution_clock::now();
            // std::cout << "Matching time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms, Matches found: " << matches.size() << std::endl;
            start = std::chrono::high_resolution_clock::now();
            cv::Mat visualizeMatches = descriptorMatcher.visualizeMatches(frame, corners, shiftedFrame, corners2, matches);
            end = std::chrono::high_resolution_clock::now();
            // std::cout << "Visualization time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;

            std::vector<geometry::Correspondence> correspondences;

            for (const auto &match : matches)
            {
                correspondences.push_back(geometry::Correspondence{
                    corners[match.queryIdx].position,
                    corners2[match.trainIdx].position});
            }
            geometry::RansacResult ransacResult = ransacHomography.estimate(correspondences);
            std::cout
                << "\nMatches: "
                << correspondences.size()
                << "\nInliers: "
                << ransacResult.inliers.size()
                << std::endl;
            std::cout << ransacResult.homography << std::endl;
            vision::Visualizer visualizer;
            // visualizer.display(visualizedCorners, "FAST Corners - Level " + std::to_string(i));
            // visualizer.display(visualizeDescriptors, "ORB Descriptors - Level " + std::to_string(i));
            visualizer.display(visualizeMatches, "Matched Features - Level " + std::to_string(i), 1280, 720);

            // visualizer.display(ransacResult.homography, "Estimated Homography - Level " + std::to_string(i));
        }
        // std::vector<vision::KeyPoint> fastCorners = vision::FastCornerDetector().detect(gray);
        // cv::Mat visualizedFastCorners = vision::FastCornerDetector().visualizeCorners(frame, fastCorners);
        // vision::Visualizer visualizer;
        // // visualizer.display(result.thresholded, "Thresholded");
        // // visualizer.display(result.edges, "Edges");
        // // visualizer.display(harrisResponse, "Harris Response");
        // // visualizer.display(visualizedHarrisCorners, "Harris Corners");
        // visualizer.display(visualizedFastCorners, "FAST Corners");

        // cv::imshow("Camera", result.edges);

        int key = cv::waitKey(1);
        if (key == 27)
        {
            break;
        }
    }
    cap.release();

    cv::destroyAllWindows();
    return 0;
}
