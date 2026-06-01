#include "vision/fast.hpp"
#include <iostream>

namespace
{
    size_t rejected = 0;
    size_t fullTests = 0;
}

bool vision::FastCornerDetector::passesHighSpeedTest(const cv::Mat &image, int x, int y) const
{
    const int center = image.at<uchar>(y, x);
    int brighter = 0;
    int darker = 0;
    const int upper = center + m_threshold;
    const int lower = center - m_threshold;

    for (int i = 0; i < 16; i += 4)
    {
        const cv::Point &offset = circle[i];
        const int value =
            image.at<uchar>(
                y + offset.y,
                x + offset.x);

        if (value > upper)
        {
            ++brighter;
        }
        else if (value < lower)
        {
            ++darker;
        }

        if (brighter >= 3 || darker >= 3)
        {
            return true;
        }
    }
    return false;
}

bool vision::FastCornerDetector::isCorner(const cv::Mat &image, int x, int y) const
{
    if (!passesHighSpeedTest(image, x, y))
    {
        rejected++;
        return false;
    }
    else
    {
        fullTests++;
    }

    const int center = image.at<uchar>(y, x);
    std::array<int, 16> state{};
    for (int i = 0; i < 16; ++i)
    {
        const cv::Point &offset = circle[i];

        const int value =
            image.at<uchar>(
                y + offset.y,
                x + offset.x);

        if (value > center + m_threshold)
        {
            state[i] = 1;
        }
        else if (value < center - m_threshold)
        {
            state[i] = -1;
        }
        else
        {
            state[i] = 0;
        }
    }
    std::array<int, 32> extended{};
    for (int i = 0; i < 32; ++i)
    {
        extended[i] = state[i % 16];
    }
    int brightRun = 0;

    for (int i = 0; i < 32; ++i)
    {
        if (extended[i] == 1)
        {
            ++brightRun;

            if (brightRun >= 9)
            {
                return true;
            }
        }
        else
        {
            brightRun = 0;
        }
    }

    int darkRun = 0;
    for (int i = 0; i < 32; ++i)
    {
        if (extended[i] == -1)
        {
            ++darkRun;

            if (darkRun >= 9)
            {
                return true;
            }
        }
        else
        {
            darkRun = 0;
        }
    }
    return false;
}

int vision::FastCornerDetector::computeScore(const cv::Mat &image, int x, int y) const
{
    const int center = image.at<uchar>(y, x);
    std::array<int, 16> diff, state;
    int score = 0;
    for (int i = 0; i < 16; ++i)
    {
        const cv::Point &offset = circle[i];
        const int value =
            image.at<uchar>(
                y + offset.y,
                x + offset.x);
        diff[i] = std::abs(value - center);
        state[i] = (value > center + m_threshold) ? 1 : (value < center - m_threshold) ? -1
                                                                                       : 0;
    }
    int bestScore = 0;

    for (int start = 0; start < 16; ++start)
    {
        int sign = state[start];

        if (sign == 0)
        {
            continue;
        }

        int arcScore = INT_MAX;
        bool valid = true;

        for (int k = 0; k < 9; ++k)
        {
            int idx = (start + k) % 16;

            if (state[idx] != sign)
            {
                valid = false;
                break;
            }

            arcScore = std::min(arcScore, diff[idx]);
        }

        if (valid)
        {
            bestScore = std::max(bestScore, arcScore);
        }
    }

    return bestScore;
}

std::vector<vision::KeyPoint> vision::FastCornerDetector::nonMaximumSuppression(const cv::Mat &scoreMap) const
{
    std::vector<vision::KeyPoint> nmsCorners;
    for (int y = 1; y < scoreMap.rows - 1; ++y)
    {
        for (int x = 1; x < scoreMap.cols - 1; ++x)
        {
            float score = scoreMap.at<int>(y, x);
            if (score == 0.0)
                continue;
            bool isMax = true;
            for (int ky = -1; ky <= 1 && isMax; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    if (ky == 0 && kx == 0)
                        continue;
                    if (scoreMap.at<int>(y + ky, x + kx) > score)
                    {
                        isMax = false;
                        break;
                    }
                }
            }
            if (isMax)
            {
                KeyPoint kp;
                kp.position = cv::Point(x, y);
                kp.score = score;
                nmsCorners.push_back(kp);
            }
        }
    }
    return nmsCorners;
}

std::vector<vision::KeyPoint> vision::FastCornerDetector::detect(const cv::Mat &image)
{
    CV_Assert(image.type() == CV_8UC1);

    rejected = 0;
    fullTests = 0;
    size_t totalPixelsExamined = 0;

    std::vector<vision::KeyPoint> corners;

    for (int y = 3; y < image.rows - 3; ++y)
    {
        for (int x = 3; x < image.cols - 3; ++x)
        {
            ++totalPixelsExamined;

            if (isCorner(image, x, y))
            {
                KeyPoint kp;
                kp.position = cv::Point(x, y);
                kp.score = computeScore(image, x, y);

                corners.push_back(kp);
            }
        }
    }
    cv::Mat scoreMap(image.rows, image.cols, CV_32SC1, cv::Scalar(0));
    for (const auto &corner : corners)
    {
        scoreMap.at<int>(corner.position) = static_cast<int>(corner.score);
    }
    std::vector<KeyPoint> nmsCorners = nonMaximumSuppression(scoreMap);
    // std::cout
    //     << "Total pixels examined: " << totalPixelsExamined
    //     << "\nRejected by quick test: " << rejected
    //     << "\nFull FAST evaluations: " << fullTests
    //     << "\nDetected corners: " << corners.size()
    //     << "\nNMS corners: " << nmsCorners.size()
    //     << std::endl;

    return nmsCorners;
}

cv::Mat vision::FastCornerDetector::visualizeCorners(const cv::Mat &image,
                                                     const std::vector<vision::KeyPoint> &corners)
{
    cv::Mat output;
    if (image.channels() == 1)
    {
        cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    }
    else
    {
        output = image.clone();
    }
    cv::Mat scoreMap(image.rows, image.cols, CV_32SC1, cv::Scalar(0));
    for (const auto &corner : corners)
    {
        cv::circle(output, corner.position, 3, cv::Scalar(0, 0, 255),
                   1, cv::LINE_AA);
        scoreMap.at<int>(corner.position) = static_cast<int>(corner.score);
    }
    return output;
}
