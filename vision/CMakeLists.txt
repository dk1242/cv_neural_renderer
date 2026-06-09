add_library(vision STATIC)

target_sources(vision
    PRIVATE
    src/convolution.cpp
    src/gaussian.cpp
    src/sobel.cpp
    src/canny.cpp
    src/visualizer.cpp
    src/harris.cpp
    src/fast.cpp
    src/image_pyramid.cpp
    src/orb.cpp
    src/descriptor_matcher.cpp
)

target_include_directories(vision
    PUBLIC
    include
)

target_link_libraries(vision
    PUBLIC
    ${OpenCV_LIBS}
)
