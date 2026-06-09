#include "vision/descriptor_matcher.h"

int vision::DescriptorMatcher::hammingDistance(const vision::OrbDescriptor &desc1, const vision::OrbDescriptor &desc2) const
{
    int distance = 0;
    for (size_t i = 0; i < desc1.size(); ++i)
    {
        distance += __builtin_popcount(desc1[i] ^ desc2[i]);
    }
    return distance;
}

std::vector<vision::Match> vision::DescriptorMatcher::match(
    const std::vector<vision::OrbDescriptor> &descriptors1,
    const std::vector<vision::OrbDescriptor> &descriptors2) const
{
    std::vector<vision::Match> matches;
    for (size_t i = 0; i < descriptors1.size(); ++i)
    {
        int bestDistance = INT_MAX;
        int bestIdx = -1;
        for (size_t j = 0; j < descriptors2.size(); ++j)
        {
            int distance = hammingDistance(descriptors1[i], descriptors2[j]);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestIdx = j;
            }
        }
        if (bestIdx != -1 && bestDistance < 50)
        {
            matches.push_back({static_cast<int>(i), bestIdx, bestDistance});
        }
    }
    std::sort(matches.begin(), matches.end(), [](const Match &a, const Match &b)
              { return a.distance < b.distance; });
    // int sameIndexMatches = 0;

    // for (const auto &match : matches)
    // {
    //     if (match.queryIdx == match.trainIdx)
    //     {
    //         ++sameIndexMatches;
    //     }
    // }

    // std::cout << "sameIndexMatches: " << sameIndexMatches << ", Matches.size(): " << matches.size() << std::endl;
    return matches;
}

std::vector<vision::Match> vision::DescriptorMatcher::crosscheckMatch(
    const std::vector<vision::Match> &forwardMatches,
    const std::vector<vision::Match> &backwardMatches) const
{
    size_t lookupSize = 0;

    for (const auto &m : backwardMatches)
    {
        if (m.queryIdx >= 0)
        {
            lookupSize = std::max(lookupSize, static_cast<size_t>(m.queryIdx) + 1);
        }
    }

    for (const auto &m : forwardMatches)
    {
        if (m.trainIdx >= 0)
        {
            lookupSize = std::max(lookupSize, static_cast<size_t>(m.trainIdx) + 1);
        }
    }

    std::vector<int> backwardLookup(lookupSize, -1);

    for (const auto &match : backwardMatches)
    {
        if (match.queryIdx >= 0)
        {
            backwardLookup[static_cast<size_t>(match.queryIdx)] = match.trainIdx;
        }
    }

    std::vector<vision::Match> crossCheckedMatches;

    for (const auto &fwdMatch : forwardMatches)
    {
        if (fwdMatch.trainIdx >= 0 &&
            backwardLookup[static_cast<size_t>(fwdMatch.trainIdx)] == fwdMatch.queryIdx)
        {
            crossCheckedMatches.push_back(fwdMatch);
        }
    }

    return crossCheckedMatches;
}

std::vector<vision::Match> vision::DescriptorMatcher::crosscheckMatch(
    const std::vector<vision::OrbDescriptor> &descriptors1,
    const std::vector<vision::OrbDescriptor> &descriptors2) const
{
    auto knnForward = knnMatch(descriptors1, descriptors2, 2);
    auto ratioForwardMatches = ratioTest(knnForward, 0.75f);
    auto knnBackward = knnMatch(descriptors2, descriptors1, 2);
    auto ratioBackwardMatches = ratioTest(knnBackward, 0.75f);
    auto crossCheckedMatches = crosscheckMatch(ratioForwardMatches, ratioBackwardMatches);
    return crossCheckedMatches;
}

std::vector<std::array<vision::Match, 2>> vision::DescriptorMatcher::knnMatch(
    const std::vector<vision::OrbDescriptor> &descriptorsA,
    const std::vector<vision::OrbDescriptor> &descriptorsB, int k) const
{
    int bestDistance = INT_MAX;
    int secondDistance = INT_MAX;

    int bestIndex = -1;
    int secondIndex = -1;
    std::vector<std::array<vision::Match, 2>> knnMatches;
    for (size_t i = 0; i < descriptorsA.size(); ++i)
    {
        bestDistance = INT_MAX;
        secondDistance = INT_MAX;
        bestIndex = -1;
        secondIndex = -1;
        for (size_t j = 0; j < descriptorsB.size(); ++j)
        {
            int distance = hammingDistance(descriptorsA[i], descriptorsB[j]);
            if (distance < bestDistance)
            {
                secondDistance = bestDistance;
                secondIndex = bestIndex;
                bestDistance = distance;
                bestIndex = j;
            }
            else if (distance < secondDistance)
            {
                secondDistance = distance;
                secondIndex = j;
            }
        }
        if (bestIndex != -1 && secondIndex != -1)
        {
            const int queryIndex = static_cast<int>(i);
            knnMatches.push_back({vision::Match{queryIndex, bestIndex, bestDistance}, vision::Match{queryIndex, secondIndex, secondDistance}});
        }
    }
    return knnMatches;
}

std::vector<vision::Match> vision::DescriptorMatcher::ratioTest(
    const std::vector<std::array<vision::Match, 2>> &knnMatches,
    float threshold) const
{
    std::vector<vision::Match> filteredMatches;
    for (const auto &pair : knnMatches)
    {
        const auto &bestMatch = pair[0];
        const auto &secondBestMatch = pair[1];
        float ratio = static_cast<float>(bestMatch.distance) / secondBestMatch.distance;

        if (ratio < threshold)
        {
            filteredMatches.push_back(bestMatch);
        }
    }
    return filteredMatches;
}

cv::Mat vision::DescriptorMatcher::visualizeMatches(
    const cv::Mat &image1, const std::vector<vision::KeyPoint> &keypoints1,
    const cv::Mat &image2, const std::vector<vision::KeyPoint> &keypoints2,
    const std::vector<vision::Match> &matches) const
{
    cv::Mat out;
    cv::hconcat(image1, image2, out);
    for (const auto &match : matches)
    {
        const auto &kp1 = keypoints1[match.queryIdx];
        const auto &kp2 = keypoints2[match.trainIdx];
        cv::Point2f pt1(static_cast<float>(kp1.position.x),
                        static_cast<float>(kp1.position.y));
        cv::Point2f pt2(static_cast<float>(kp2.position.x),
                        static_cast<float>(kp2.position.y));
        pt2 += cv::Point2f(static_cast<float>(image1.cols), 0.0f);
        cv::line(out, pt1, pt2, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    return out;
}
