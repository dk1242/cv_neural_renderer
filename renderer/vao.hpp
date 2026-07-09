#pragma once

namespace renderer
{

class VBO;

class VAO
{
public:
    VAO();
    ~VAO();

    VAO(const VAO&) = delete;
    VAO& operator=(const VAO&) = delete;

    void bind() const;
    void unbind() const;
    void linkAttrib(const VBO& vbo, unsigned int layout, int numComponents) const;

private:
    unsigned int id_ = 0;
};

}
