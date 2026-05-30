#include "vision/canny.hpp"

#include <queue>

constexpr uchar NO_EDGE = 0;
constexpr uchar WEAK_EDGE = 1;
constexpr uchar STRONG_EDGE = 2;

cv::Mat vision::CannyDetector::nonMaximumSuppression(const cv::Mat &magnitude,
                                                     const cv::Mat &angle)
{
    CV_Assert(magnitude.type() == CV_32FC1);
    CV_Assert(angle.type() == CV_32FC1);
    CV_Assert(magnitude.size() == angle.size());

    cv::Mat output = cv::Mat::zeros(magnitude.size(), CV_32FC1);

    for (int y = 1; y < magnitude.rows - 1; ++y)
    {
        for (int x = 1; x < magnitude.cols - 1; ++x)
        {
            float theta = angle.at<float>(y, x);

            // Convert angle to [0, 180)
            if (theta < 0)
                theta += 180;

            float current = magnitude.at<float>(y, x);

            float neighbor1 = 0.0f;
            float neighbor2 = 0.0f;

            // 0 degree
            if ((theta >= 0 && theta < 22.5f) ||
                (theta >= 157.5f && theta <= 180.0f))
            {
                neighbor1 = magnitude.at<float>(y, x - 1);

                neighbor2 = magnitude.at<float>(y, x + 1);
            }

            // 45 degree
            else if (theta >= 22.5f && theta < 67.5f)
            {
                neighbor1 = magnitude.at<float>(y - 1, x + 1);

                neighbor2 = magnitude.at<float>(y + 1, x - 1);
            }

            // 90 degree
            else if (theta >= 67.5f && theta < 112.5f)
            {
                neighbor1 = magnitude.at<float>(y - 1, x);

                neighbor2 = magnitude.at<float>(y + 1, x);
            }

            // 135 degree
            else
            {
                neighbor1 = magnitude.at<float>(y - 1, x - 1);

                neighbor2 = magnitude.at<float>(y + 1, x + 1);
            }

            if (current >= neighbor1 && current >= neighbor2)
            {
                output.at<float>(y, x) = current;
            }
            else
            {
                output.at<float>(y, x) = 0.0f;
            }
        }
    }
    return output;
}

cv::Mat vision::CannyDetector::doubleThreshold(const cv::Mat &nms,
                                               float lowThreshold,
                                               float highThreshold)
{
    CV_Assert(nms.type() == CV_32FC1);
    CV_Assert(lowThreshold >= 0);
    CV_Assert(lowThreshold < highThreshold);

    cv::Mat output(
        nms.rows,
        nms.cols,
        CV_8UC1,
        cv::Scalar(NO_EDGE));

    for (int y = 0; y < nms.rows; ++y)
    {
        for (int x = 0; x < nms.cols; ++x)
        {
            float value = nms.at<float>(y, x);

            if (value >= highThreshold)
            {
                output.at<uchar>(y, x) = STRONG_EDGE;
            }
            else if (value >= lowThreshold)
            {
                output.at<uchar>(y, x) = WEAK_EDGE;
            }
            else
            {
                output.at<uchar>(y, x) = NO_EDGE;
            }
        }
    }
    return output;
}
cv::Mat vision::CannyDetector::hysteresis(const cv::Mat &thresholded)
{
    CV_Assert(thresholded.type() == CV_8UC1);

    cv::Mat output = thresholded.clone();

    std::queue<cv::Point> q;

    // Push all strong pixels
    for (int y = 0; y < output.rows; ++y)
    {
        for (int x = 1; x < output.cols - 1; ++x)
        {
            if (output.at<uchar>(y, x) == STRONG_EDGE)
            {
                q.push(cv::Point(x, y));
            }
        }
    }

    // 8-connected neighbors
    int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};

    while (!q.empty())
    {
        cv::Point p = q.front();
        q.pop();

        for (int i = 0; i < 8; ++i)
        {
            int nx = p.x + dx[i];
            int ny = p.y + dy[i];

            if (nx < 0 || ny < 0 ||
                nx >= output.cols ||
                ny >= output.rows)
            {
                continue;
            }

            uchar &neighbor = output.at<uchar>(ny, nx);

            // Promote weak -> strong
            if (neighbor == WEAK_EDGE)
            {
                neighbor = STRONG_EDGE;
                q.push(cv::Point(nx, ny));
            }
        }
    }

    // Remove remaining weak pixels
    for (int y = 0; y < output.rows; ++y)
    {
        for (int x = 0; x < output.cols; ++x)
        {
            uchar &pixel =
                output.at<uchar>(y, x);

            if (pixel != STRONG_EDGE)
            {
                pixel = NO_EDGE;
            }
        }
    }
    return output;
}

cv::Mat vision::CannyDetector::detectEdges(
    const cv::Mat &image,
    float lowThreshold,
    float highThreshold)
{
    CV_Assert(image.type() == CV_8UC1);

    cv::Mat edges = detectDebug(image, lowThreshold, highThreshold).edges * 127;

    return edges;
}

vision::CannyResult vision::CannyDetector::detectDebug(const cv::Mat &image, float lowThreshold, float highThreshold)
{
    vision::CannyResult result;

    // 1. Noise reduction
    result.blurred = m_gaussianFilter.apply(image);

    // 2. Gradients
    result.gx = m_sobelFilter.computeX(result.blurred);
    result.gy = m_sobelFilter.computeY(result.blurred);

    // 3. Magnitude + direction
    result.magnitude = m_sobelFilter.magnitude(result.gx, result.gy);
    result.direction = m_sobelFilter.direction(result.gx, result.gy);

    // 4. Edge thinning
    result.nms = nonMaximumSuppression(result.magnitude,
                                       result.direction);

    // 5. Edge classification
    result.thresholded = doubleThreshold(result.nms,
                                         lowThreshold,
                                         highThreshold);

    // 6. Edge linking
    result.edges = hysteresis(result.thresholded);

    return result;
}
