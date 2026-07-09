find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

find_path(GLM_INCLUDE_DIR NAMES glm/glm.hpp HINTS /usr/include /usr/local/include)
if(NOT GLM_INCLUDE_DIR)
    message(FATAL_ERROR "GLM not found — install glm-devel (or equivalent).")
endif()

add_library(renderer STATIC
    camera.cpp
    fly_camera.cpp
    geometry_object.cpp
    opengl_renderer.cpp
    shader.cpp
    vao.cpp
    vbo.cpp
    glad/glad.c
)

target_include_directories(renderer
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/glad
    ${GLM_INCLUDE_DIR}
)

target_compile_definitions(renderer
    PRIVATE
    RENDERER_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders"
    PUBLIC
    GLM_ENABLE_EXPERIMENTAL
)

target_link_libraries(renderer
    PUBLIC
    ${OpenCV_LIBS}
    OpenGL::GL
    glfw
)
