cmake_minimum_required(VERSION 3.20)

project(
    VisionEngine
    VERSION 0.1
    LANGUAGES C CXX
)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install path" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_subdirectory(core)
add_subdirectory(vision)
add_subdirectory(geometry)
add_subdirectory(slam)
add_subdirectory(renderer)
add_subdirectory(app)

install(TARGETS VisionEngineApp
    RUNTIME DESTINATION bin
)
