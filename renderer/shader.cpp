#include "shader.hpp"

#include <fstream>
#include <iostream>
#include <sstream>

namespace
{
    std::string readFile(const char* path)
    {
        std::ifstream file(path);
        if (!file)
        {
            std::cerr << "Shader: failed to open " << path << std::endl;
            return {};
        }

        std::ostringstream stream;
        stream << file.rdbuf();
        return stream.str();
    }
}

namespace renderer
{

Shader::Shader(const char* vertexFilePath, const char* fragmentFilePath, const char* shaderName_)
    : shaderName(shaderName_)
{
    const std::string vertexCode = readFile(vertexFilePath);
    const std::string fragmentCode = readFile(fragmentFilePath);
    const char* vertexSource = vertexCode.c_str();
    const char* fragmentSource = fragmentCode.c_str();

    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, nullptr);
    glCompileShader(vertexShader);
    compileErrors(vertexShader, "VERTEX");

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, nullptr);
    glCompileShader(fragmentShader);
    compileErrors(fragmentShader, "FRAGMENT");

    ID = glCreateProgram();
    glAttachShader(ID, vertexShader);
    glAttachShader(ID, fragmentShader);
    glLinkProgram(ID);
    compileErrors(ID, "PROGRAM");

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

Shader::Shader(const char* computeShaderFilePath, const char* shaderName_)
    : shaderName(shaderName_)
{
    const std::string computeCode = readFile(computeShaderFilePath);
    const char* computeSource = computeCode.c_str();

    GLuint computeShader = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(computeShader, 1, &computeSource, nullptr);
    glCompileShader(computeShader);
    compileErrors(computeShader, "COMPUTE");

    ID = glCreateProgram();
    glAttachShader(ID, computeShader);
    glLinkProgram(ID);
    compileErrors(ID, "PROGRAM");

    glDeleteShader(computeShader);
}

void Shader::Activate() const
{
    glUseProgram(ID);
}

void Shader::Delete() const
{
    glDeleteProgram(ID);
}

void Shader::compileErrors(unsigned int shader, const char* type)
{
    GLint success = 0;
    GLchar infoLog[1024];

    if (std::string(type) != "PROGRAM")
    {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success)
        {
            glGetShaderInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Shader compile error (" << shaderName << ", " << type << "): " << infoLog << std::endl;
        }
    }
    else
    {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if (!success)
        {
            glGetProgramInfoLog(shader, sizeof(infoLog), nullptr, infoLog);
            std::cerr << "Shader link error (" << shaderName << "): " << infoLog << std::endl;
        }
    }
}

}
