add_executable(VisionEngineApp
    main.cpp
)

target_link_libraries(
    VisionEngineApp
    PRIVATE
    vision
    geometry
)
