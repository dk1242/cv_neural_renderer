#include "vision/orb.hpp"
#include <algorithm>
#include <bitset>
#include <random>
#include <iostream>

vision::OrbExtractor::OrbExtractor()
{
    for (int i = 0; i < 256; ++i)
    {
        vision::OrbExtractor::PointPair pair;
        pair.p = cv::Point2f(bit_pattern_31_[4 * i], bit_pattern_31_[4 * i + 1]);
        pair.q = cv::Point2f(bit_pattern_31_[4 * i + 2], bit_pattern_31_[4 * i + 3]);

        m_pattern.push_back(pair);
    }
}

float vision::OrbExtractor::computeOrientation(const cv::Mat &image, const vision::KeyPoint &kp) const
{
    float m00 = 0.0f, m10 = 0.0f, m01 = 0.0f;
    constexpr int radius = 15;

    for (int y = -radius; y <= radius; ++y)
    {
        for (int x = -radius; x <= radius; ++x)
        {
            if (x * x + y * y > radius * radius)
                continue;

            int pixelValue = image.at<uchar>(kp.position.y + y, kp.position.x + x);
            m00 += pixelValue;
            m10 += x * pixelValue;
            m01 += y * pixelValue;
        }
    }
    if (m00 == 0.0f)
    {
        return 0.0f;
    }
    float cx = m10 / m00;
    float cy = m01 / m00;

    return std::atan2(cy, cx);
}

cv::Point2f vision::OrbExtractor::rotatePoint(const cv::Point2f &point, float cosA, float sinA) const
{
    return cv::Point2f(
        point.x * cosA - point.y * sinA,
        point.x * sinA + point.y * cosA);
}

uint8_t vision::OrbExtractor::sampleIntensity(const cv::Mat &image, const cv::Point2f &point) const
{
    int x0 = static_cast<int>(std::floor(point.x));
    int y0 = static_cast<int>(std::floor(point.y));

    int x1 = x0 + 1;
    int y1 = y0 + 1;

    if (x0 < 0 || x1 >= image.cols ||
        y0 < 0 || y1 >= image.rows)
    {
        return 0;
    }

    float dx = point.x - x0;
    float dy = point.y - y0;

    float I00 = image.at<uchar>(y0, x0);
    float I10 = image.at<uchar>(y0, x1);
    float I01 = image.at<uchar>(y1, x0);
    float I11 = image.at<uchar>(y1, x1);

    float value = (1.0f - dx) * (1.0f - dy) * I00 +
                  dx * (1.0f - dy) * I10 +
                  (1.0f - dx) * dy * I01 +
                  dx * dy * I11;
    return static_cast<uint8_t>(value);
}

std::array<uint8_t, 32> vision::OrbExtractor::computeDescriptor(const cv::Mat &image, const KeyPoint &kp) const
{
    std::array<uint8_t, 32> descriptor = {0};
    float angleDeg = kp.angle * 180.0f / CV_PI;

    angleDeg = std::round(angleDeg / 12.0f) * 12.0f;

    float angleRad = angleDeg * CV_PI / 180.0f;
    // float angleRad = kp.angle;
    float cosA = std::cos(angleRad);
    float sinA = std::sin(angleRad);
    for (size_t i = 0; i < m_pattern.size(); ++i)
    {
        cv::Point2f rotatedP = rotatePoint(m_pattern[i].p, cosA, sinA);
        cv::Point2f rotatedQ = rotatePoint(m_pattern[i].q, cosA, sinA);

        cv::Point2f center(static_cast<float>(kp.position.x),
                           static_cast<float>(kp.position.y));
        cv::Point2f sampleP = center + rotatedP;
        cv::Point2f sampleQ = center + rotatedQ;

        uint8_t intensityP = sampleIntensity(image, sampleP);
        uint8_t intensityQ = sampleIntensity(image, sampleQ);

        if (intensityP < intensityQ)
        {
            descriptor[i / 8] |= (1 << (i % 8));
        }
    }
    // std::cout<<"descriptor size "<<sizeof(descriptor)<<std::endl;
    return descriptor;
}

void vision::OrbExtractor::computeOrientations(const cv::Mat &image, std::vector<vision::KeyPoint> &keypoints) const
{
    for (auto &kp : keypoints)
    {
        if (kp.position.x < 15 ||
            kp.position.x >= image.cols - 15 ||
            kp.position.y < 15 ||
            kp.position.y >= image.rows - 15)
        {
            continue;
        }
        kp.angle = computeOrientation(image, kp);
    }
}

std::vector<vision::OrbDescriptor> vision::OrbExtractor::computeDescriptors(
    const cv::Mat &image,
    const std::vector<vision::KeyPoint> &keypoints) const
{
    std::vector<vision::OrbDescriptor> descriptors;
    for (size_t i = 0; i < keypoints.size(); ++i)
    {
        auto descriptor = computeDescriptor(image, keypoints[i]);
        descriptors.push_back(descriptor);
    }
    return descriptors;
}

cv::Mat vision::OrbExtractor::visualizeArrows(const cv::Mat &image, const std::vector<vision::KeyPoint> &keypoints) const
{
    cv::Mat out;
    if (image.channels() == 1)
        cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);
    else
        out = image.clone();

    for (const auto &kp : keypoints)
    {
        float angleRad = kp.angle;
        cv::Point2f end(kp.position.x + 20.0f * std::cos(angleRad),
                        kp.position.y + 20.0f * std::sin(angleRad));
        cv::arrowedLine(out, kp.position, end, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    return out;
}

cv::Mat vision::OrbExtractor::visualizeDescriptorPattern(const cv::Mat &image, const std::vector<vision::KeyPoint> &keypoints) const
{
    cv::Mat out;
    if (image.channels() == 1)
        cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);
    else
        out = image.clone();
    for (const auto &kp : keypoints)
    {
        float angleRad = kp.angle;
        float cosA = std::cos(angleRad);
        float sinA = std::sin(angleRad);
        for (size_t i = 0; i < m_pattern.size(); ++i)
        {
            cv::Point2f rotatedP = rotatePoint(m_pattern[i].p, cosA, sinA);
            cv::Point2f rotatedQ = rotatePoint(m_pattern[i].q, cosA, sinA);

            cv::Point2f center(static_cast<float>(kp.position.x),
                               static_cast<float>(kp.position.y));
            cv::Point2f sampleP = center + rotatedP;
            cv::Point2f sampleQ = center + rotatedQ;

            // cv::circle(out, sampleP, 2, cv::Scalar(0, 255, 0), -1);
            // cv::circle(out, sampleQ, 2, cv::Scalar(0, 0, 255), -1);
            cv::line(out, sampleP, sampleQ, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        }
    }
    return out;
}
