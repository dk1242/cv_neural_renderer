#include "vision/visualizer.h"

namespace
{
    bool isDisplayReady(const cv::Mat &image)
    {
        const int channels = image.channels();

        if (image.depth() != CV_8U ||
            (channels != 1 && channels != 3 && channels != 4))
        {
            return false;
        }

        if (channels == 1)
        {
            double minValue = 0.0;
            double maxValue = 0.0;
            cv::minMaxLoc(image, &minValue, &maxValue);

            return maxValue > 2.0;
        }

        return true;
    }
}

cv::Mat vision::Visualizer::normalizeToDisplay(const cv::Mat &image)
{
    cv::Mat normalized;
    cv::normalize(
        image,
        normalized,
        0,
        255,
        cv::NORM_MINMAX);

    normalized.convertTo(
        normalized,
        CV_8UC1);

    return normalized;
}

void vision::Visualizer::display(const cv::Mat &image, const std::string &windowName)
{
    const cv::Mat displayImage =
        isDisplayReady(image) ? image : normalizeToDisplay(image);
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 640, 720);
    cv::imshow(windowName, displayImage);
}
