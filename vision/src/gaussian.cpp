#include "vision/gaussian.hpp"

cv::Mat vision::GaussianFilter::apply(const cv::Mat &image)
{
    CV_Assert(image.channels() == 1);
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_32FC1);

    int kernel[5] = {1, 4, 6, 4, 1};
    int kernelSum = 16;

    cv::Mat temp(image.rows, image.cols, CV_32FC1, cv::Scalar::all(0));
    cv::Mat output(image.rows, image.cols, CV_32FC1, cv::Scalar::all(0));

    const bool isFloat = image.type() == CV_32FC1;

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 2; x < image.cols - 2; x++)
        {
            float s = 0.0f;
            for (int k = -2; k <= 2; k++)
            {
                float p = isFloat ? image.at<float>(y, x + k)
                                  : static_cast<float>(image.at<uchar>(y, x + k));
                int weight = kernel[k + 2];
                s += p * weight;
            }
            temp.at<float>(y, x) = s / kernelSum;
        }
    }

    for (int y = 2; y < image.rows - 2; y++)
    {
        for (int x = 2; x < image.cols - 2; x++)
        {
            float s = 0.0f;
            for (int k = -2; k <= 2; k++)
            {
                float p = temp.at<float>(y + k, x);
                int weight = kernel[k + 2];
                s += p * weight;
            }
            s /= kernelSum;
            output.at<float>(y, x) = s;
        }
    }

    // if (image.type() == CV_8UC1)
    // {
    //     cv::Mat output8u;
    //     output.convertTo(output8u, CV_8UC1);
    //     return output8u;
    // }

    return output;
}
