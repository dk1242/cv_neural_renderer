#include "vision/convolution.hpp"

cv::Mat vision::ConvolutionFilter::apply(const cv::Mat &image, const cv::Mat &kernel)
{
    CV_Assert(image.type() == CV_8UC1 || image.type() == CV_32FC1);
    CV_Assert(kernel.type() == CV_32FC1);

    CV_Assert(kernel.rows % 2 == 1);
    CV_Assert(kernel.cols % 2 == 1);

    const int halfRows = kernel.rows / 2;
    const int halfCols = kernel.cols / 2;

    cv::Mat output(image.rows, image.cols, CV_32FC1, cv::Scalar(0));

    for (int y = halfRows; y < image.rows - halfRows; ++y)
    {
        for (int x = halfCols; x < image.cols - halfCols; ++x)
        {
            float sum = 0.0f;

            for (int ky = -halfRows; ky <= halfRows; ++ky)
            {
                for (int kx = -halfCols; kx <= halfCols; ++kx)
                {
                    float pixel;
                    if (image.depth() == CV_8U)
                    {
                        pixel = static_cast<float>(image.at<uchar>(y + ky,
                                                                   x + kx));
                    }
                    else
                    {
                        pixel = image.at<float>(y + ky, x + kx);
                    }
                    const float weight = kernel.at<float>(ky + halfRows,
                                                          kx + halfCols);

                    sum += pixel * weight;
                }
            }
            output.at<float>(y, x) = sum;
        }
    }
    return output;
}