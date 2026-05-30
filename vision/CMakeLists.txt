add_library(vision STATIC)

target_sources(vision
    PRIVATE
    src/gaussian.cpp
    src/sobel.cpp
    src/canny.cpp
)

target_include_directories(vision
    PUBLIC
    include
)

target_link_libraries(vision
    PUBLIC
    ${OpenCV_LIBS}
)