#include "fly_camera.hpp"

#include <cmath>

namespace renderer
{

FlyCamera::FlyCamera(int width, int height, glm::vec3 position)
    : width(width), height(height)
{
    Position = position;
    near = 0.1f;
    far = 1000.0f;
    UpdateFrustum();
    updateMatrix();
}

void FlyCamera::updateMatrix()
{
    view = glm::lookAt(Position, Position + cameraOrientation, Up);
    projection = glm::perspective(glm::radians(FOVdeg), static_cast<float>(width) / static_cast<float>(height), near, far);
    cameraMatrix = projection * view;
}

void FlyCamera::UpdateFrustum()
{
    const float aspect = static_cast<float>(width) / static_cast<float>(height);
    const float halfHeight = near * tanf(glm::radians(FOVdeg) / 2.0f);
    const float halfWidth = halfHeight * aspect;

    top = halfHeight;
    bottom = -halfHeight;
    right = halfWidth;
    left = -halfWidth;
}

void FlyCamera::Matrix(Shader& shader, const char* uniform)
{
    glUniformMatrix4fv(glGetUniformLocation(shader.ID, uniform), 1, GL_FALSE, glm::value_ptr(cameraMatrix));
}

void FlyCamera::Inputs(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        Position += speed * playerOrientation;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        Position += speed * -glm::normalize(glm::cross(playerOrientation, Up));
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        Position += speed * -playerOrientation;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        Position += speed * glm::normalize(glm::cross(playerOrientation, Up));
    }
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
    {
        Position += speed * Up;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS)
    {
        Position += speed * -Up;
    }
    speed = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ? 2.0f : 0.1f;

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS)
    {
        // GLFW_CURSOR_DISABLED (rather than HIDDEN + glfwSetCursorPos re-centering)
        // gives virtual, unbounded cursor deltas on platforms that can't warp the
        // pointer (e.g. Wayland), so we never call glfwSetCursorPos here.
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        double mouseX = 0.0;
        double mouseY = 0.0;
        glfwGetCursorPos(window, &mouseX, &mouseY);

        if (firstClick)
        {
            lastMouseX_ = mouseX;
            lastMouseY_ = mouseY;
            firstClick = false;
        }

        const float rotX = sensitivity * static_cast<float>(mouseY - lastMouseY_) / static_cast<float>(height);
        const float rotY = sensitivity * static_cast<float>(mouseX - lastMouseX_) / static_cast<float>(width);

        lastMouseX_ = mouseX;
        lastMouseY_ = mouseY;

        const glm::vec3 newCameraOrientation = glm::rotate(cameraOrientation, glm::radians(-rotX), glm::normalize(glm::cross(cameraOrientation, Up)));
        if (std::abs(glm::angle(newCameraOrientation, Up) - glm::radians(90.0f)) <= glm::radians(85.0f))
        {
            cameraOrientation = newCameraOrientation;
        }

        playerOrientation = glm::rotate(playerOrientation, glm::radians(-rotY), Up);
        cameraOrientation = glm::rotate(cameraOrientation, glm::radians(-rotY), Up);

        playerOrientation = glm::normalize(playerOrientation);
        cameraOrientation = glm::normalize(cameraOrientation);
    }
    else if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE)
    {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        firstClick = true;
    }
}

}
