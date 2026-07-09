#pragma once

#include <cstddef>

namespace renderer
{

class VBO
{
public:
    VBO();
    ~VBO();

    VBO(const VBO&) = delete;
    VBO& operator=(const VBO&) = delete;

    void bind() const;
    void unbind() const;
    void setData(const float* data, std::size_t sizeBytes) const;

    unsigned int id() const { return id_; }

private:
    unsigned int id_ = 0;
};

}
