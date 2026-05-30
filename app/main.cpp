#include "vision/gaussian.hpp"
#include "vision/sobel.hpp"
#include "vision/visualizer.h"
#include "vision/canny.hpp"
#include "vision/harris.hpp"

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
        vision::HarrisCornerDetector harrisDetector;
        cv::Mat harrisResponse = harrisDetector.computeResponse(gray);
        std::vector<cv::Point> corners = harrisDetector.detectCorners(harrisResponse, 0.01f);
        cv::Mat visualizedCorners = harrisDetector.visualizeCorners(frame, corners);

        vision::Visualizer visualizer;
        // visualizer.display(result.thresholded, "Thresholded");
        // visualizer.display(result.edges, "Edges");
        visualizer.display(harrisResponse, "Harris Response");
        visualizer.display(visualizedCorners, "Harris Corners");

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
