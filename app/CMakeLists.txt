add_executable(VisionEngineApp
    main.cpp
)

target_compile_definitions(
    VisionEngineApp
    PRIVATE
    VISION_ENGINE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(
    VisionEngineApp
    PRIVATE
    vision
    geometry
    slam
    renderer
)
