#include "vision/gaussian.hpp"

cv::Mat vision::GaussianFilter::apply(const cv::Mat &image)
{
    CV_Assert(image.channels() == 3);

    int kernel[5] = {1, 4, 6, 4, 1};
    int kernelSum = 16;

    // temp holds intermediate horizontal pass results as 3-channel float
    cv::Mat temp(image.rows, image.cols, CV_32FC3, cv::Scalar::all(0));
    cv::Mat output(image.rows, image.cols, CV_8UC3, cv::Scalar::all(0));

    // Horizontal pass (per-channel)
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 2; x < image.cols - 2; x++)
        {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
            for (int k = -2; k <= 2; k++)
            {
                cv::Vec3b p = image.at<cv::Vec3b>(y, x + k);
                int weight = kernel[k + 2];
                s0 += p[0] * weight;
                s1 += p[1] * weight;
                s2 += p[2] * weight;
            }
            temp.at<cv::Vec3f>(y, x) = cv::Vec3f(s0 / kernelSum, s1 / kernelSum, s2 / kernelSum);
        }
    }

    // Vertical pass (per-channel)
    for (int y = 2; y < image.rows - 2; y++)
    {
        for (int x = 2; x < image.cols - 2; x++)
        {
            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f;
            for (int k = -2; k <= 2; k++)
            {
                cv::Vec3f p = temp.at<cv::Vec3f>(y + k, x);
                int weight = kernel[k + 2];
                s0 += p[0] * weight;
                s1 += p[1] * weight;
                s2 += p[2] * weight;
            }
            s0 /= kernelSum;
            s1 /= kernelSum;
            s2 /= kernelSum;
            output.at<cv::Vec3b>(y, x) = cv::Vec3b(static_cast<uchar>(s0), static_cast<uchar>(s1), static_cast<uchar>(s2));
        }
    }
    return output;
}