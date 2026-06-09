#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "orb.hpp"

namespace vision
{
    class DescriptorMatcher
    {
        int hammingDistance(const vision::OrbDescriptor &desc1, const vision::OrbDescriptor &desc2) const;

    public:
        std::vector<vision::Match> match(const std::vector<vision::OrbDescriptor> &descriptors1,
                                         const std::vector<vision::OrbDescriptor> &descriptors2) const;
        std::vector<vision::Match> crosscheckMatch(const std::vector<vision::OrbDescriptor> &descriptors1,
                                                   const std::vector<vision::OrbDescriptor> &descriptors2) const;
        std::vector<std::array<Match, 2>> knnMatch(const std::vector<vision::OrbDescriptor> &descriptorsA,
                                                   const std::vector<vision::OrbDescriptor> &descriptorsB, int k = 2) const;
        std::vector<Match> ratioTest(const std::vector<std::array<Match, 2>> &knnMatches,
                                     float threshold) const;
        std::vector<Match> crosscheckMatch(const std::vector<Match> &forwardMatches,
                                           const std::vector<Match> &backwardMatches) const;

        cv::Mat visualizeMatches(const cv::Mat &image1, const std::vector<vision::KeyPoint> &keypoints1,
                                 const cv::Mat &image2, const std::vector<vision::KeyPoint> &keypoints2,
                                 const std::vector<vision::Match> &matches) const;
    };
}
