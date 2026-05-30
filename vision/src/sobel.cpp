#include "vision/sobel.hpp"

cv::Mat vision::SobelFilter::computeX(const cv::Mat &image)
{
    CV_Assert(image.type() == CV_8UC1);
    static const cv::Mat kernel = (cv::Mat_<float>(3, 3) << -1, 0, 1,
                                   -2, 0, 2,
                                   -1, 0, 1);

    return m_convolution.apply(
        image,
        kernel);
}

cv::Mat vision::SobelFilter::computeY(const cv::Mat &image)
{
    CV_Assert(image.type() == CV_8UC1);
    static const cv::Mat kernel =
        (cv::Mat_<float>(3, 3) << -1, -2, -1,
         0, 0, 0,
         1, 2, 1);

    return m_convolution.apply(
        image,
        kernel);
}

cv::Mat vision::SobelFilter::magnitude(const cv::Mat &gx, const cv::Mat &gy)
{
    CV_Assert(gx.type() == CV_32FC1);
    CV_Assert(gy.type() == CV_32FC1);
    CV_Assert(gx.size() == gy.size());

    cv::Mat mag(gx.size(), CV_32FC1);
    for (int y = 0; y < gx.rows; ++y)
    {
        for (int x = 0; x < gx.cols; ++x)
        {
            float gxVal = gx.at<float>(y, x);
            float gyVal = gy.at<float>(y, x);
            mag.at<float>(y, x) = std::sqrt(gxVal * gxVal + gyVal * gyVal);
        }
    }
    return mag;
}

cv::Mat vision::SobelFilter::direction(const cv::Mat &gx, const cv::Mat &gy)
{
    CV_Assert(gx.type() == CV_32FC1);
    CV_Assert(gy.type() == CV_32FC1);
    CV_Assert(gx.size() == gy.size());

    cv::Mat dir(gx.size(), CV_32FC1);
    for (int y = 0; y < gx.rows; ++y)
    {
        for (int x = 0; x < gx.cols; ++x)
        {
            float gxVal = gx.at<float>(y, x);
            float gyVal = gy.at<float>(y, x);
            dir.at<float>(y, x) = std::atan2(gyVal, gxVal);
        }
    }
    return dir;
}