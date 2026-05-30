#include "vision/harris.hpp"
#include <opencv2/opencv.hpp>

cv::Mat vision::HarrisCornerDetector::computeResponse(const cv::Mat &image)
{
    CV_Assert(image.type() == CV_8UC1);

    cv::Mat Ix = m_sobelFilter.computeX(image);
    cv::Mat Iy = m_sobelFilter.computeY(image);

    cv::Mat Ix2, Iy2, Ixy;
    cv::multiply(Ix, Ix, Ix2);
    cv::multiply(Iy, Iy, Iy2);
    cv::multiply(Ix, Iy, Ixy);

    Ix2 = m_gaussianFilter.apply(Ix2);
    Iy2 = m_gaussianFilter.apply(Iy2);
    Ixy = m_gaussianFilter.apply(Ixy);

    cv::Mat harrisResponse(image.size(), CV_32F);

    for (int y = 0; y < image.rows; ++y)
    {
        const float *pIx2 = Ix2.ptr<float>(y);
        const float *pIxy = Ixy.ptr<float>(y);
        const float *pIy2 = Iy2.ptr<float>(y);
        float *pR = harrisResponse.ptr<float>(y);
        for (int x = 0; x < image.cols; ++x)
        {
            float a = pIx2[x];
            float b = pIxy[x];
            float c = pIy2[x];
            float det = a * c - b * b;
            float trace = a + c;
            pR[x] = det - k * trace * trace;
        }
    }
    return harrisResponse;
}

std::vector<cv::Point> vision::HarrisCornerDetector::detectCorners(
    const cv::Mat &harrisResponse, float thresholdFactor)
{
    CV_Assert(harrisResponse.type() == CV_32FC1);

    double minVal, maxVal;
    cv::minMaxLoc(harrisResponse, &minVal, &maxVal);

    const float responseThreshold = static_cast<float>(maxVal) *
                                    thresholdFactor;

    std::vector<cv::Point> corners;

    for (int y = 1; y < harrisResponse.rows - 1; ++y)
    {
        for (int x = 1; x < harrisResponse.cols - 1; ++x)
        {
            const float val = harrisResponse.at<float>(y, x);

            if (val <= responseThreshold)
                continue;

            bool isMaximum = true;

            for (int ky = -1; ky <= 1 && isMaximum; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    if (ky == 0 && kx == 0)
                        continue;

                    if (harrisResponse.at<float>(
                            y + ky,
                            x + kx) > val)
                    {
                        isMaximum = false;
                        break;
                    }
                }
            }

            if (isMaximum)
            {
                corners.emplace_back(x, y);
            }
        }
    }
    return corners;
}

cv::Mat vision::HarrisCornerDetector::visualizeCorners(
    const cv::Mat &image,
    const std::vector<cv::Point> &corners)
{
    cv::Mat output;
    if(image.channels() == 1)
    {
        cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    }
    else
    {
        output = image.clone();
    }

    for (const auto &corner : corners)
    {
        cv::circle(output, corner, 3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    }
    return output;
}
