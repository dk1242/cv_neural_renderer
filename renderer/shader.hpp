#pragma once

#include <string>

namespace renderer
{

class Shader
{
public:
    void load(const std::string& vertexPath,
              const std::string& fragmentPath);
    void use() const;
};

}
