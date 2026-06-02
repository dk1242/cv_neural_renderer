#include "vision/gaussian.hpp"
#include "vision/sobel.hpp"
#include "vision/visualizer.h"
#include "vision/canny.hpp"
#include "vision/harris.hpp"
#include "vision/fast.hpp"
#include "vision/image_pyramid.h"
#include "vision/orb.hpp"

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
            std::vector<vision::KeyPoint> corners = vision::FastCornerDetector().detect(level.image);
            for (auto &kp : corners)
                kp.octave = i;
            // cv::Mat visualizedCorners = vision::FastCornerDetector().visualizeCorners(level.image, corners);
            orbExtractor.computeOrientations(level.image, corners);
            cv::Mat visualizedCorners = orbExtractor.visualizeArrows(frame, corners);
            cv::Mat descriptors = orbExtractor.visualizeDescriptorPattern(frame, corners);
            vision::Visualizer visualizer;
            visualizer.display(visualizedCorners, "FAST Corners - Level " + std::to_string(i));
            visualizer.display(descriptors, "ORB Descriptors - Level " + std::to_string(i));
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
