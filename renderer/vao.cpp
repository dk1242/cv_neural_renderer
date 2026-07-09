#include "vao.hpp"
#include "vbo.hpp"

#include "glad/glad.h"

namespace renderer
{

VAO::VAO()
{
    glGenVertexArrays(1, &id_);
}

VAO::~VAO()
{
    glDeleteVertexArrays(1, &id_);
}

void VAO::bind() const
{
    glBindVertexArray(id_);
}

void VAO::unbind() const
{
    glBindVertexArray(0);
}

void VAO::linkAttrib(const VBO& vbo, unsigned int layout, int numComponents) const
{
    bind();
    vbo.bind();
    glVertexAttribPointer(layout, numComponents, GL_FLOAT, GL_FALSE,
                           numComponents * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(layout);
    unbind();
}

}
