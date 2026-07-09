#pragma once

#include "glad/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <glm/gtx/vector_angle.hpp>

#include "shader.hpp"

namespace renderer
{

class FlyCamera
{
public:
    glm::vec3 Position = glm::vec3(0.0f, 0.0f, 10.0f);
    glm::vec3 cameraOrientation = glm::vec3(0.0f, 0.0f, -1.0f); // look direction, affected by pitch
    glm::vec3 playerOrientation = glm::vec3(0.0f, 0.0f, -1.0f); // movement direction, yaw only
    glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f);                 // y is up

    glm::mat4 view;
    glm::mat4 projection;
    glm::mat4 cameraMatrix = glm::mat4(1.0f);

    int width;
    int height;

    float left, right, top, bottom, near, far;

    float speed = 0.1f;

    FlyCamera(int width, int height, glm::vec3 position);

    // Updates the camera matrix
    void updateMatrix();
    void UpdateFrustum();
    // Exports the camera matrix to shader
    void Matrix(Shader& shader, const char* uniform);
    // Handles camera inputs
    void Inputs(GLFWwindow* window);

    float fovDegrees() const { return FOVdeg; }

private:
    float FOVdeg = 45.0f;
    float sensitivity = 100.0f;
    bool firstClick = true;
    double lastMouseX_ = 0.0;
    double lastMouseY_ = 0.0;
};

}
