add_library(geometry STATIC)

target_sources(geometry
    PRIVATE
    homography.cpp
    ransac.cpp
    epipolar_geometry.cpp
    normalization.cpp
    fundamental_matrix.cpp
    essential_matrix.cpp
    pose_recovery.cpp
    triangulation.cpp
)

target_include_directories(geometry
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(geometry
    PUBLIC
    ${OpenCV_LIBS}
)
