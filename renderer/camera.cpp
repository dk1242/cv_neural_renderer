#include "camera.hpp"

namespace renderer
{

void Camera::setPose(const cv::Mat& pose)
{
    pose.copyTo(pose_);
}

const cv::Mat& Camera::pose() const
{
    return pose_;
}

}
