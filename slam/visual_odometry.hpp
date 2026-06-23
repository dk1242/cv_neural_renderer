#pragma once

#include <opencv2/opencv.hpp>
#include "vision/fast.hpp"
#include "vision/orb.hpp"
#include "geometry/epipolar_geometry.hpp"

namespace slam
{
    enum class VOState
    {
        FirstFrame,
        Bootstrap,
        Tracking
    };
    struct Pose
    {
        cv::Mat Rwc;
        cv::Mat twc;
    };
    struct Frame
    {
        cv::Mat image;

        std::vector<vision::KeyPoint> keypoints;

        cv::Mat descriptors;

        std::vector<cv::Point2f> points;
    };

    class VisualOdometry
    {
    public:
        void processFrame(const cv::Mat &frame);

    private:
        slam::VOState state_;

        Frame previousFrame_;

        slam::Pose currentPose_;

        std::vector<slam::Pose> trajectory_;
    };
}
