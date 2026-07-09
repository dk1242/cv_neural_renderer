#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

struct GLFWwindow;

#include "camera.hpp"
#include "fly_camera.hpp"
#include "geometry_object.hpp"
#include "shader.hpp"

namespace renderer
{

class OpenGLRenderer
{
public:
    OpenGLRenderer();
    ~OpenGLRenderer();

    bool initialize(int width = 1024, int height = 768, const std::string& title = "VisionEngine 3D View");
    void setCamera(const Camera& camera);
    void setMapPoints(const std::vector<cv::Point3d>& points);
    void renderFrame();
    bool shouldClose() const;
    void pollEvents();
    void cleanup();

private:
    GLFWwindow* window_ = nullptr;
    int width_ = 0;
    int height_ = 0;
    Camera camera_;
    std::vector<cv::Point3d> mapPoints_;
    bool initialized_ = false;

    std::unique_ptr<Shader> shader_;
    std::unique_ptr<GeometryObject> pointsGeometry_;
    std::unique_ptr<GeometryObject> gridGeometry_;
    std::unique_ptr<GeometryObject> axesGeometry_;
    float gridHalfExtent_ = 20.0f;
    std::unique_ptr<FlyCamera> flyCamera_;

    bool createWindow(int width, int height, const std::string& title);
    bool loadOpenGL();
    bool setupShaderAndBuffers();
    void rebuildGrid(float halfExtent);
    void frameCameraOnMapPoints();
    static void errorCallback(int error, const char* description);
};

}
