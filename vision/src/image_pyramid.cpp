#include "vision/image_pyramid.h"
#include <opencv2/opencv.hpp>

cv::Mat vision::ImagePyramid::downsample(const cv::Mat &image) const
{
    cv::Mat downsampled((image.rows + 1) / 2, (image.cols + 1) / 2, image.type());
    for (int y = 0; y < image.rows; y += 2)
    {
        for (int x = 0; x < image.cols; x += 2)
        {
            downsampled.at<uchar>(y / 2, x / 2) = image.at<uchar>(y, x);
            // image.row(y).col(x).copyTo(downsampled.row(y / 2).col(x / 2));
        }
    }
    return downsampled;
}

std::vector<vision::Level> vision::ImagePyramid::build(const cv::Mat &image, int numLevels, float scaleFactor) const
{
    std::vector<vision::Level> levels;
    levels.reserve(numLevels);

    GaussianFilter gaussian;

    for (int level = 0; level < numLevels; ++level)
    {
        float scale = 1.0f / pow(scaleFactor, level);
        cv::Mat resized;

        cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_LINEAR);

        cv::Mat blurred = gaussian.apply(resized);

        blurred.convertTo(blurred, image.type());

        levels.push_back({blurred, scale});
    }

    return levels;
}
