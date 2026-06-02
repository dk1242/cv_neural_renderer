#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "fast.hpp"

namespace vision
{
    using OrbDescriptor = std::array<uint8_t, 32>;
    struct Match
    {
        int queryIdx;
        int trainIdx;
        int distance;
    };
    class OrbExtractor
    {
        struct PointPair
        {
            cv::Point2f p;
            cv::Point2f q;
        };

        std::vector<PointPair> m_pattern;

        float computeOrientation(const cv::Mat &image, const vision::KeyPoint &kp) const;
        cv::Point2f rotatePoint(const cv::Point2f &point, float cosA, float sinA) const;
        uint8_t sampleIntensity(const cv::Mat &image, const cv::Point2f &point) const;
        vision::OrbDescriptor computeDescriptor(const cv::Mat &image, const KeyPoint &kp) const;

    public:
        OrbExtractor();
        void computeOrientations(
            const cv::Mat &image,
            std::vector<vision::KeyPoint> &keypoints) const;
        std::vector<vision::OrbDescriptor> computeDescriptors(const cv::Mat &image,
                                                      const std::vector<vision::KeyPoint> &keypoints) const;

        cv::Mat visualizeArrows(const cv::Mat &image,
                                const std::vector<vision::KeyPoint> &keypoints) const;
        cv::Mat visualizeDescriptorPattern(const cv::Mat &image,
                                           const std::vector<vision::KeyPoint> &keypoints) const;
    };

}
