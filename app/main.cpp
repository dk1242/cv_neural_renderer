#include "vision/gaussian.hpp"

int main()
{
    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cout << "Cannot open camera\n";
        return -1;
    }

    cv::namedWindow("Camera", cv::WINDOW_NORMAL);
    cv::resizeWindow("Camera", 1280, 720);

    cv::Mat frame, blurred;

    while (true)
    {
        cap >> frame;

        if (frame.empty())
        {
            break;
        }

        blurred = vision::GaussianFilter().apply(frame);
        cv::imshow("Camera", blurred);

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
