#pragma once

#include <cstddef>
#include <glm/glm.hpp>

#include "vao.hpp"
#include "vbo.hpp"

namespace renderer
{

// Bundles the VAO/VBO pair + draw parameters shared by every plain
// vertex-color line/point set the renderer draws (points cloud, grid, axes),
// so call sites don't each repeat bind/upload/draw boilerplate.
class GeometryObject
{
public:
    explicit GeometryObject(unsigned int drawMode, int componentsPerVertex = 3);

    void setData(const float* vertices, std::size_t sizeBytes);
    void draw(unsigned int shaderId, const glm::vec3& color) const;
    void draw(unsigned int shaderId, const glm::vec3& color, int first, int count) const;

    int vertexCount() const { return vertexCount_; }

private:
    VAO vao_;
    VBO vbo_;
    unsigned int drawMode_;
    int componentsPerVertex_;
    int vertexCount_ = 0;
};

}
