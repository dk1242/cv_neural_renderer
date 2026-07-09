#include "vbo.hpp"

#include "glad/glad.h"

namespace renderer
{

VBO::VBO()
{
    glGenBuffers(1, &id_);
}

VBO::~VBO()
{
    glDeleteBuffers(1, &id_);
}

void VBO::bind() const
{
    glBindBuffer(GL_ARRAY_BUFFER, id_);
}

void VBO::unbind() const
{
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void VBO::setData(const float* data, std::size_t sizeBytes) const
{
    bind();
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(sizeBytes), data, GL_DYNAMIC_DRAW);
    unbind();
}

}
