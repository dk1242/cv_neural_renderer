# OpenGL 4.3+ Project Setup Template

A reusable checklist + boilerplate for bootstrapping a new OpenGL 4.3+ (GLFW + GLAD + GLM + stb) project on Linux with CMake. Based on the structure of this project. Targets 4.3 specifically because that's the minimum version exposing **compute shaders**, **SSBOs**, and **`glDebugMessageCallback`** — all used once a project grows past basic rendering (e.g. GPU-side culling/LOD, indirect draws).

## 1. Dependencies

Install via system package manager (Fedora example — swap for apt/pacman as needed):

```bash
sudo dnf install cmake gcc-c++ glfw-devel glm-devel mesa-libGL-devel
```

- **GLFW** — window/context/input. Installed as a system package (`glfw3` via pkg-config).
- **GLM** — math library (header-only).
- **GLAD** — OpenGL function loader. Generate at https://glad.dav1d.de with **gl 4.3 (or higher), Core profile, C/C++**, and drop the generated `include/glad/`, `include/KHR/`, and `src/glad.c` into the project. Picking 3.3 here is the most common reason "it compiles but `glDispatchCompute`/SSBO calls don't exist" errors show up later — the loader simply won't have generated those symbols.
- **stb** — header-only image loading (`stb_image.h`) from https://github.com/nothings/stb. Put it in a `stb/` folder.
- **mesa-libGL-devel** — only provides the `GL/gl.h` headers and the generic `libGL.so` link stub needed to *compile*. It does not decide which GPU renders at runtime.

### Running on NVIDIA specifically

Install the proprietary NVIDIA driver (`akmod-nvidia` / `nvidia-driver` depending on distro) — this is what actually renders, not the mesa devel package. At runtime, `libGL.so` is resolved through libglvnd's vendor dispatch, which picks the active driver. On hybrid (Optimus/PRIME) laptops with both Intel/AMD integrated and NVIDIA discrete GPUs, force the NVIDIA GPU explicitly with:

```bash
env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia ./MyGLProject
```

`run.sh` below already sets these env vars before launching. Verify it worked with `glxinfo | grep "OpenGL renderer"` while those vars are set — it should print the NVIDIA GPU name, not `llvmpipe` or the integrated GPU.

## 2. Project layout

```
project-root/
├── CMakeLists.txt
├── run.sh                 # configure + build + run helper
├── main.cpp
├── ShaderClass.h/.cpp      # shader compile/link wrapper
├── Camera.h/.cpp           # fly camera (view/projection, input handling)
├── Texture.h/.cpp          # texture loading via stb
├── VAO.h/.cpp  VBO.h/.cpp  EBO.h/.cpp   # buffer wrappers
├── glad.c                  # generated loader source
├── stb/stb_image.h
├── stb.cpp                 # #define STB_IMAGE_IMPLEMENTATION + include
├── *.vert / *.frag / *.comp  # shader source files (copied next to binary at build time)
└── textures/                # runtime assets (copied to build dir)
```

## 3. CMakeLists.txt (minimal working template)

```cmake
cmake_minimum_required(VERSION 3.16)
project(MyGLProject VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW3 REQUIRED glfw3)

find_path(GLM_INCLUDE_DIR NAMES glm/glm.hpp HINTS /usr/include /usr/local/include)
find_path(GLAD_INCLUDE_DIR NAMES glad/glad.h HINTS /usr/include /usr/local/include ${CMAKE_CURRENT_SOURCE_DIR})
find_path(STB_INCLUDE_DIR NAMES stb/stb_image.h HINTS ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT GLM_INCLUDE_DIR)
    message(FATAL_ERROR "GLM not found.")
endif()
if(NOT GLAD_INCLUDE_DIR)
    message(FATAL_ERROR "GLAD not found — generate it and place include/glad in the project or system include path.")
endif()
if(NOT STB_INCLUDE_DIR)
    message(FATAL_ERROR "stb_image.h not found — clone stb into ./stb.")
endif()

set(SOURCE_FILES
    main.cpp
    Camera.cpp
    ShaderClass.cpp
    Texture.cpp
    VAO.cpp
    VBO.cpp
    EBO.cpp
    SSBO.cpp          # only needed once you add compute shaders — see §6.1
    glad.c
    stb.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLFW3_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIR}
    ${GLAD_INCLUDE_DIR}
    ${STB_INCLUDE_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE GLFW_INCLUDE_NONE GLM_ENABLE_EXPERIMENTAL)

target_link_libraries(${PROJECT_NAME} PRIVATE
    OpenGL::GL
    ${GLFW3_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# copy shaders next to the binary so relative paths in code just work
file(GLOB SHADER_FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/*.vert"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.frag"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.comp"
)
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_FILES} $<TARGET_FILE_DIR:${PROJECT_NAME}>
)

# copy runtime assets (optional)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/textures")
    file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/textures" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()
```

> Add `find_package(OpenMP REQUIRED)` + link `OpenMP::OpenMP_CXX` only if you actually parallelize CPU-side work (e.g. per-instance culling), like the source project does.

## 4. run.sh helper

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT_DIR/build"

echo "Configuring project (out-of-source build)..."
cmake -S "$ROOT_DIR" -B "$BUILD_DIR"

echo "Building project..."
cmake --build "$BUILD_DIR" -- -j"$(nproc)"

EXE="$BUILD_DIR/MyGLProject"
[ -x "$EXE" ] || { echo "Error: executable not found: $EXE" >&2; exit 1; }

# On hybrid-GPU laptops, force the discrete NVIDIA GPU via PRIME offload:
env __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia "$EXE"
```

```bash
chmod +x run.sh
./run.sh
```

## 5. Shader wrapper (ShaderClass.h/.cpp)

Loads vertex/fragment (or compute) source from disk, compiles, links, and reports errors. Core pattern:

```cpp
// ShaderClass.h
class Shader {
public:
    GLuint ID;
    Shader(const char* vertexPath, const char* fragmentPath, const char* name);
    Shader(const char* computePath, const char* name);      // compute-only overload
    void Activate() const;
    void Delete() const;
private:
    void compileErrors(unsigned int shader, const char* type);
};
```

Implementation: read file to `std::string` → `glCreateShader` → `glShaderSource` → `glCompileShader` → check `GL_COMPILE_STATUS` → attach to a `glCreateProgram()` → `glLinkProgram` → check `GL_LINK_STATUS` → `glDeleteShader` the intermediates.

Every shader source file must start with `#version 430 core` (or higher) to match the 4.3 context requested in `main.cpp` — mismatched versions are a common silent-failure source (GLSL falls back to defaults instead of erroring in some drivers).

### Compute shaders (new in 4.3)

```cpp
Shader cullShader("cull.comp", "cull");
cullShader.Activate();
glDispatchCompute(numGroupsX, 1, 1);
glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT); // sync before the SSBO is read by the next draw
```

Pair with an SSBO wrapper (`SSBO.h/.cpp`) alongside VAO/VBO/EBO — same `Bind`/`Unbind`/`Delete` shape, but binds to `GL_SHADER_STORAGE_BUFFER` and an indexed binding point (`glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, ID)`) so both the compute shader and the vertex shader can see the same buffer.

## 6. Camera (Camera.h/.cpp)

Fly-style camera holding `Position`, `Orientation`, `Up`, and derived `view`/`projection`/`cameraMatrix`. Exposes:
- `updateMatrix()` — rebuilds `view` (glm::lookAt) and `projection` (glm::perspective), combines into `cameraMatrix`.
- `Matrix(Shader&, uniformName)` — uploads `cameraMatrix` via `glUniformMatrix4fv`.
- `Inputs(GLFWwindow*)` — WASD + mouse-look, called once per frame before draw calls.

## 7. Buffer wrappers (VAO/VBO/EBO/SSBO)

Thin RAII-ish wrappers around `glGenVertexArrays/Buffers`, `glBindBuffer`, `glBufferData`, and `glVertexAttribPointer`, each with a `Bind()`/`Unbind()`/`Delete()`. Keep them tiny — one class per GL object type. Add an `SSBO` wrapper (same shape, binds to `GL_SHADER_STORAGE_BUFFER`) and an instanced-VBO wrapper once you're using compute shaders or instanced rendering — see §6.1.

## 8. Texture loading (Texture.h/.cpp)

Wrap `stbi_load` → `glGenTextures` → `glTexImage2D` → `glGenerateMipmap` → `stbi_image_free`. Remember:
```cpp
// stb.cpp — exactly one .cpp in the whole project defines this
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
```

## 9. main.cpp skeleton

```cpp
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "ShaderClass.h"
#include "Camera.h"

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifndef NDEBUG
    glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE); // enables glDebugMessageCallback below
#endif

    GLFWwindow* window = glfwCreateWindow(1280, 720, "MyGLProject", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);

    gladLoadGL();
    glViewport(0, 0, 1280, 720);

#ifndef NDEBUG
    // 4.3+ debug output — routes GL errors/warnings straight to your console
    // instead of silent failures or manual glGetError() polling.
    glEnable(GL_DEBUG_OUTPUT);
    glDebugMessageCallback([](GLenum, GLenum, GLuint, GLenum severity,
                              GLsizei, const GLchar* message, const void*) {
        std::cerr << "[GL DEBUG] " << message << std::endl;
    }, nullptr);
#endif

    Shader shaderProgram("main.vert", "main.frag", "main");
    Camera camera(1280, 720, glm::vec3(0.0f, 0.0f, 10.0f));

    while (!glfwWindowShouldClose(window)) {
        glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        camera.Inputs(window);
        camera.updateMatrix();
        shaderProgram.Activate();
        camera.Matrix(shaderProgram, "camMatrix");

        // draw calls here

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    shaderProgram.Delete();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
```

## 10. Quick-start checklist for a new project

1. `mkdir project && cd project && git init`
2. Copy `CMakeLists.txt` + `run.sh` templates above, rename `MyGLProject`.
3. Generate GLAD (Core, gl4.3+) → drop `glad.c` + `include/` in.
4. Clone/copy `stb_image.h` into `stb/`, add `stb.cpp`.
5. Copy `ShaderClass.h/.cpp`, `Camera.h/.cpp`, `VAO/VBO/EBO`, `Texture.h/.cpp` from a prior project (or write from the skeletons above).
6. Write `main.vert` / `main.frag`.
7. `chmod +x run.sh && ./run.sh`.
8. Add a `.gitignore` excluding `build/`.
