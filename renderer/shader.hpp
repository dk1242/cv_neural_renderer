#pragma once

#include "glad/glad.h"

#include <string>

namespace renderer
{

class Shader
{
public:
    GLuint ID = 0;
    std::string shaderName;

    Shader(const char* vertexFilePath, const char* fragmentFilePath, const char* shaderName);
    Shader(const char* computeShaderFilePath, const char* shaderName);

    void Activate() const;
    void Delete() const;

private:
    void compileErrors(unsigned int shader, const char* type);
};

}
