add_library(slam STATIC)

target_sources(slam
    PRIVATE
    visual_odometry.cpp
    feature_track.cpp
    track_manager.cpp
    map.cpp
    mapping.cpp
)

target_include_directories(slam
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(slam
    PUBLIC
    ${OpenCV_LIBS}
    vision
    geometry
)
