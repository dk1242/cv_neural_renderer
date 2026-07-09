#include "geometry_object.hpp"

#include "glad/glad.h"

namespace renderer
{

GeometryObject::GeometryObject(unsigned int drawMode, int componentsPerVertex)
    : drawMode_(drawMode), componentsPerVertex_(componentsPerVertex)
{
    vao_.linkAttrib(vbo_, 0, componentsPerVertex_);
}

void GeometryObject::setData(const float* vertices, std::size_t sizeBytes)
{
    vbo_.setData(vertices, sizeBytes);
    vertexCount_ = static_cast<int>(sizeBytes / (componentsPerVertex_ * sizeof(float)));
}

void GeometryObject::draw(unsigned int shaderId, const glm::vec3& color) const
{
    draw(shaderId, color, 0, vertexCount_);
}

void GeometryObject::draw(unsigned int shaderId, const glm::vec3& color, int first, int count) const
{
    if (count <= 0)
    {
        return;
    }

    glUniform3f(glGetUniformLocation(shaderId, "uColor"), color.r, color.g, color.b);
    vao_.bind();
    glDrawArrays(drawMode_, first, count);
    vao_.unbind();
}

}
