#include <algorithm>
#include <cctype>
#include <iostream>
#include <filesystem>
#include <vector>

#include <opencv2/opencv.hpp>

#include "vision/visualizer.h"
#include "slam/visual_odometry.hpp"
#include "renderer/opengl_renderer.hpp"
#include "renderer/camera.hpp"

static std::vector<std::filesystem::path> loadImagePaths(const std::filesystem::path &folder)
{
    std::vector<std::filesystem::path> imagePaths;
    if (!std::filesystem::is_directory(folder))
    {
        return imagePaths;
    }

    for (const auto &entry : std::filesystem::directory_iterator(folder))
    {
        if (!entry.is_regular_file())
        {
            continue;
        }

        std::string extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c)
                       { return static_cast<char>(std::tolower(c)); });

        if (extension == ".png" || extension == ".jpg" || extension == ".jpeg" || extension == ".bmp" || extension == ".tiff" || extension == ".tif")
        {
            imagePaths.push_back(entry.path());
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    return imagePaths;
}

static cv::Mat buildViewPoseFromCameraPose(const slam::CameraPose &cameraPose)
{
    cv::Mat viewPose = cv::Mat::eye(4, 4, CV_64F);
    if (cameraPose.Rwc.empty() || cameraPose.twc.empty())
    {
        viewPose.at<double>(2, 3) = -5.0;
        return viewPose;
    }

    cv::Mat Rcw = cameraPose.Rwc.t();
    cv::Mat t = -Rcw * cameraPose.twc;
    Rcw.copyTo(viewPose(cv::Range(0, 3), cv::Range(0, 3)));
    t.copyTo(viewPose(cv::Range(0, 3), cv::Range(3, 4)));
    return viewPose;
}

vision::Visualizer vis;

void testVisualOdometry()
{
    const std::filesystem::path sourceFolder = "/home/dk1242/Desktop/opencv/basics/VisionEngine/datasets/drive/image_00/data";
    auto imageFiles = loadImagePaths(sourceFolder);
    if (imageFiles.empty())
    {
        std::cerr << "No image files found in '" << sourceFolder << "'" << std::endl;
        return;
    }

    std::cout << "Loaded " << imageFiles.size() << " images from " << sourceFolder << std::endl;
    slam::VisualOdometry visualOdometry;

    const size_t maxFrames = 20;
    const size_t frameCount = std::min(imageFiles.size(), maxFrames);

    for (size_t i = 0; i < frameCount; ++i)
    {
        const auto &imagePath = imageFiles[i];
        cv::Mat frame = cv::imread(imagePath.string(), cv::IMREAD_COLOR);
        if (frame.empty())
        {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            continue;
        }
        visualOdometry.processFrame(frame);
    }

    visualOdometry.setCameraCenters();

    std::vector<cv::Point3d> mapPoints;
    mapPoints.reserve(visualOdometry.mapPoints().size());
    for (const auto &point : visualOdometry.mapPoints())
    {
        mapPoints.push_back(point.position);
    }

    renderer::OpenGLRenderer openglRenderer;
    if (openglRenderer.initialize(1024, 768, "VisionEngine 3D View"))
    {
        renderer::Camera camera;
        camera.setPose(buildViewPoseFromCameraPose(visualOdometry.currentPose()));
        openglRenderer.setCamera(camera);
        openglRenderer.setMapPoints(mapPoints);

        while (!openglRenderer.shouldClose())
        {
            openglRenderer.renderFrame();
            openglRenderer.pollEvents();
        }
    }
    else
    {
        std::cerr << "Failed to initialize OpenGL renderer" << std::endl;
    }

    // cv::Mat trajectoryImage = visualOdometry.drawTrajectory();
    // vis.display(trajectoryImage, "Trajectory", 800, 800);
    // cv::waitKey(0);
}

int main(int argc, char **argv)
{
    testVisualOdometry();
    return 0;
}
