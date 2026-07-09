#include "opengl_renderer.hpp"

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include "glad/glad.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>

namespace
{
    std::array<float, 16> buildProjectionMatrix(double verticalFieldOfViewDeg,
                                                double aspectRatio,
                                                double zNear,
                                                double zFar)
    {
        const double fovRad = verticalFieldOfViewDeg * CV_PI / 180.0;
        const double f = 1.0 / std::tan(fovRad * 0.5);
        const double range = zNear - zFar;

        return {
            static_cast<float>(f / aspectRatio), 0.0f, 0.0f, 0.0f,
            0.0f, static_cast<float>(f), 0.0f, 0.0f,
            0.0f, 0.0f, static_cast<float>((zFar + zNear) / range), -1.0f,
            0.0f, 0.0f, static_cast<float>((2.0 * zFar * zNear) / range), 0.0f};
    }

    std::array<float, 16> buildViewMatrix(const cv::Mat &pose)
    {
        std::array<float, 16> view = {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, -5.0f, 1.0f};

        if (pose.empty() || pose.rows != 4 || pose.cols != 4)
        {
            return view;
        }

        cv::Mat pose64;
        if (pose.type() != CV_64F)
        {
            pose.convertTo(pose64, CV_64F);
        }
        else
        {
            pose64 = pose;
        }

        for (int row = 0; row < 4; ++row)
        {
            for (int col = 0; col < 4; ++col)
            {
                view[col * 4 + row] = static_cast<float>(pose64.at<double>(row, col));
            }
        }

        return view;
    }
}

namespace renderer
{

    OpenGLRenderer::OpenGLRenderer() = default;

    OpenGLRenderer::~OpenGLRenderer()
    {
        cleanup();
    }

    bool OpenGLRenderer::initialize(int width, int height, const std::string &title)
    {
        glfwSetErrorCallback(errorCallback);
        if (glfwInit() == GLFW_FALSE)
        {
            std::cerr << "OpenGLRenderer: failed to initialize GLFW" << std::endl;
            return false;
        }

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);

        window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
        if (!window_)
        {
            std::cerr << "OpenGLRenderer: failed to create GLFW window" << std::endl;
            glfwTerminate();
            return false;
        }

        glfwMakeContextCurrent(window_);

        // GLFW_MAXIMIZED means the requested width/height above are not what we
        // actually got — use the real framebuffer size for the viewport/aspect
        // ratio instead, or the scene renders into a small corner of a maximized
        // window while the camera projection still assumes the original size.
        glfwGetFramebufferSize(window_, &width, &height);

        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress)))
        {
            std::cerr << "OpenGLRenderer: failed to initialize GLAD" << std::endl;
            glfwDestroyWindow(window_);
            glfwTerminate();
            return false;
        }

        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

        shader_ = std::make_unique<Shader>(RENDERER_SHADER_DIR "/points.vert",
                                           RENDERER_SHADER_DIR "/points.frag",
                                           "points");
        if (shader_->ID == 0)
        {
            std::cerr << "OpenGLRenderer: failed to create shader program" << std::endl;
            glfwDestroyWindow(window_);
            glfwTerminate();
            return false;
        }

        pointsVBO_ = std::make_unique<VBO>();
        pointsVAO_ = std::make_unique<VAO>();
        pointsVAO_->linkAttrib(*pointsVBO_, 0, 3);

        gridVBO_ = std::make_unique<VBO>();
        gridVAO_ = std::make_unique<VAO>();
        gridVAO_->linkAttrib(*gridVBO_, 0, 3);
        rebuildGrid(gridHalfExtent_);

        {
            constexpr float axisLength = 2.0f;
            const float axesVertices[] = {
                0.0f, 0.0f, 0.0f, axisLength, 0.0f, 0.0f, // X
                0.0f, 0.0f, 0.0f, 0.0f, axisLength, 0.0f, // Y
                0.0f, 0.0f, 0.0f, 0.0f, 0.0f, axisLength  // Z
            };

            axesVBO_ = std::make_unique<VBO>();
            axesVBO_->setData(axesVertices, sizeof(axesVertices));
            axesVAO_ = std::make_unique<VAO>();
            axesVAO_->linkAttrib(*axesVBO_, 0, 3);
        }

        flyCamera_ = std::make_unique<FlyCamera>(width, height, glm::vec3(0.0f, 0.0f, 10.0f));

        glViewport(0, 0, width, height);
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glClearColor(0.08f, 0.08f, 0.12f, 1.0f);

        width_ = width;
        height_ = height;
        initialized_ = true;

        return true;
    }

    void OpenGLRenderer::setCamera(const Camera &camera)
    {
        camera_ = camera;

        // Seed the fly camera's position from the SLAM-estimated pose so free
        // navigation starts from where the tracked camera actually was.
        //
        // camera_.pose() stores a *view* matrix (world-to-camera: p_cam = R*p_world + t),
        // not a camera-to-world pose. The camera's actual world position is
        // -R^T * t, not the raw translation column t — using t directly (as this
        // used to) places the fly camera somewhere essentially arbitrary relative
        // to the map.
        if (flyCamera_)
        {
            const cv::Mat &pose = camera_.pose();
            if (!pose.empty() && pose.rows == 4 && pose.cols == 4)
            {
                cv::Mat pose64;
                if (pose.type() != CV_64F)
                {
                    pose.convertTo(pose64, CV_64F);
                }
                else
                {
                    pose64 = pose;
                }

                const cv::Mat R = pose64(cv::Range(0, 3), cv::Range(0, 3));
                const cv::Mat t = pose64(cv::Range(0, 3), cv::Range(3, 4));
                const cv::Mat worldPosition = -R.t() * t;

                // The SLAM pipeline (geometry/, slam/) works entirely in the standard
                // CV/pinhole convention: X-right, Y-down, Z-forward-into-the-scene
                // (cheirality checks require z > 0 in front of the camera). OpenGL's
                // convention is Y-up, with the camera looking down -Z. Flip Y and Z
                // here at the renderer boundary so "up" in the grid/fly-camera
                // actually matches "up" in the reconstructed scene.
                flyCamera_->Position = glm::vec3(
                    static_cast<float>(worldPosition.at<double>(0)),
                    static_cast<float>(-worldPosition.at<double>(1)),
                    static_cast<float>(-worldPosition.at<double>(2)));
            }
        }
    }

    void OpenGLRenderer::setMapPoints(const std::vector<cv::Point3d> &points)
    {
        // Same CV-to-GL convention flip as setCamera() above: (x, y, z) -> (x, -y, -z).
        mapPoints_.clear();
        mapPoints_.reserve(points.size());
        for (const auto &point : points)
        {
            mapPoints_.emplace_back(point.x, -point.y, -point.z);
        }

        std::vector<float> pointData;
        pointData.reserve(mapPoints_.size() * 3);
        for (const auto &point : mapPoints_)
        {
            pointData.push_back(static_cast<float>(point.x));
            pointData.push_back(static_cast<float>(point.y));
            pointData.push_back(static_cast<float>(point.z));
        }

        pointsVBO_->setData(pointData.data(), pointData.size() * sizeof(float));

        frameCameraOnMapPoints();
    }

    void OpenGLRenderer::renderFrame()
    {
        if (!window_)
        {
            return;
        }

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        flyCamera_->updateMatrix();

        shader_->Activate();
        glUniformMatrix4fv(glGetUniformLocation(shader_->ID, "uProjection"), 1, GL_FALSE, glm::value_ptr(flyCamera_->projection));
        glUniformMatrix4fv(glGetUniformLocation(shader_->ID, "uView"), 1, GL_FALSE, glm::value_ptr(flyCamera_->view));
        drawGrid();
        drawAxes();

        glUniform3f(glGetUniformLocation(shader_->ID, "uColor"), 1.0f, 1.0f, 0.6f);
        drawMapPoints();

        glUseProgram(0);
        glfwSwapBuffers(window_);
    }

    bool OpenGLRenderer::shouldClose() const
    {
        return window_ ? glfwWindowShouldClose(window_) : true;
    }

    void OpenGLRenderer::pollEvents()
    {
        if (window_)
        {
            if (flyCamera_)
            {
                flyCamera_->Inputs(window_);
            }
            glfwPollEvents();
        }
    }

    void OpenGLRenderer::cleanup()
    {
        flyCamera_.reset();
        pointsVAO_.reset();
        pointsVBO_.reset();
        gridVAO_.reset();
        gridVBO_.reset();
        axesVAO_.reset();
        axesVBO_.reset();

        if (shader_)
        {
            shader_->Delete();
            shader_.reset();
        }

        if (window_)
        {
            glfwDestroyWindow(window_);
            window_ = nullptr;
        }

        if (initialized_)
        {
            glfwTerminate();
            initialized_ = false;
        }
    }

    void OpenGLRenderer::drawMapPoints() const
    {
        if (mapPoints_.empty())
        {
            return;
        }

        pointsVAO_->bind();
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(mapPoints_.size()));
        pointsVAO_->unbind();
    }

    void OpenGLRenderer::rebuildGrid(float halfExtent)
    {
        gridHalfExtent_ = halfExtent;
        const float spacing = halfExtent / 10.0f;

        std::vector<float> gridVertices;
        for (float coord = -halfExtent; coord <= halfExtent + spacing * 0.5f; coord += spacing)
        {
            // line parallel to X axis
            gridVertices.insert(gridVertices.end(), {-halfExtent, 0.0f, coord, halfExtent, 0.0f, coord});
            // line parallel to Z axis
            gridVertices.insert(gridVertices.end(), {coord, 0.0f, -halfExtent, coord, 0.0f, halfExtent});
        }

        gridVertexCount_ = static_cast<int>(gridVertices.size() / 3);
        gridVBO_->setData(gridVertices.data(), gridVertices.size() * sizeof(float));
    }

    void OpenGLRenderer::frameCameraOnMapPoints()
    {
        if (!flyCamera_ || mapPoints_.empty())
        {
            return;
        }

        // Degenerate triangulations (near-parallel rays, noisy matches) can produce
        // a handful of non-finite or wildly-outlying points. Filter those out and
        // use a high percentile of distance-from-centroid rather than the raw
        // min/max, so a few bad points can't blow up the framing distance and push
        // the whole scene outside the far clip plane (which would render nothing).
        std::vector<cv::Point3d> finitePoints;
        finitePoints.reserve(mapPoints_.size());
        for (const auto &point : mapPoints_)
        {
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
            {
                finitePoints.push_back(point);
            }
        }
        if (finitePoints.empty())
        {
            return;
        }

        cv::Point3d centroidSum(0.0, 0.0, 0.0);
        for (const auto &point : finitePoints)
        {
            centroidSum += point;
        }
        const glm::vec3 center(
            static_cast<float>(centroidSum.x / static_cast<double>(finitePoints.size())),
            static_cast<float>(centroidSum.y / static_cast<double>(finitePoints.size())),
            static_cast<float>(centroidSum.z / static_cast<double>(finitePoints.size())));

        std::vector<float> distances;
        distances.reserve(finitePoints.size());
        for (const auto &point : finitePoints)
        {
            const glm::vec3 p(static_cast<float>(point.x), static_cast<float>(point.y), static_cast<float>(point.z));
            distances.push_back(glm::length(p - center));
        }
        const std::size_t percentileIndex = std::min(
            static_cast<std::size_t>(static_cast<double>(distances.size()) * 0.97),
            distances.size() - 1);
        std::nth_element(distances.begin(), distances.begin() + percentileIndex, distances.end());
        const float radius = std::max(distances[percentileIndex], 1.0f);

        rebuildGrid(radius * 1.5f);

        // Derive the framing distance from the actual vertical FOV instead of a
        // fixed multiplier, so a bounding sphere of this radius is guaranteed to
        // fit inside the frustum (a fixed multiplier only "happens" to work for
        // whatever shape the point cloud is; an elongated trajectory trail can
        // easily poke outside the frame otherwise). marginFactor adds breathing
        // room around the content.
        constexpr float marginFactor = 1.4f;
        const float halfFovRad = glm::radians(flyCamera_->fovDegrees()) * 0.5f;
        const float distance = (radius / std::tan(halfFovRad)) * marginFactor;

        const glm::vec3 offsetDir = glm::normalize(glm::vec3(0.35f, 0.6f, 1.0f));
        flyCamera_->Position = center + offsetDir * distance;
        // near stays a small constant, not derived from this one-time framing
        // distance — this is a free-fly camera, so the user can navigate right up
        // to (or past) any point afterward, and a near plane tied to the initial
        // distance would clip nearby geometry as soon as they got closer than it.
        // far still scales with the scene so distant content isn't clipped.
        flyCamera_->near = 0.05f;
        flyCamera_->far = (distance + radius) * 3.0f;

        const glm::vec3 lookDir = glm::normalize(center - flyCamera_->Position);
        flyCamera_->cameraOrientation = lookDir;
        const glm::vec3 flatLookDir(lookDir.x, 0.0f, lookDir.z);
        flyCamera_->playerOrientation = glm::length(flatLookDir) > 1e-4f ? glm::normalize(flatLookDir) : glm::vec3(0.0f, 0.0f, -1.0f);
    }

    void OpenGLRenderer::drawGrid() const
    {
        if (!gridVAO_)
        {
            return;
        }

        glUniform3f(glGetUniformLocation(shader_->ID, "uColor"), 0.35f, 0.35f, 0.4f);
        gridVAO_->bind();
        glDrawArrays(GL_LINES, 0, gridVertexCount_);
        gridVAO_->unbind();
    }

    void OpenGLRenderer::drawAxes() const
    {
        if (!axesVAO_)
        {
            return;
        }

        axesVAO_->bind();

        glUniform3f(glGetUniformLocation(shader_->ID, "uColor"), 1.0f, 0.0f, 0.0f); // X - red
        glDrawArrays(GL_LINES, 0, 2);

        glUniform3f(glGetUniformLocation(shader_->ID, "uColor"), 0.0f, 1.0f, 0.0f); // Y - green
        glDrawArrays(GL_LINES, 2, 2);

        glUniform3f(glGetUniformLocation(shader_->ID, "uColor"), 0.0f, 0.4f, 1.0f); // Z - blue
        glDrawArrays(GL_LINES, 4, 2);

        axesVAO_->unbind();
    }

    void OpenGLRenderer::errorCallback(int error, const char *description)
    {
        std::cerr << "GLFW error [" << error << "]: " << description << std::endl;
    }
}