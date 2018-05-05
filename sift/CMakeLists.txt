cmake_minimum_required(VERSION 2.8)
project( sift )
find_package(OpenCV REQUIRED )

include_directories(include)
set(SOURCES src/sift.cpp)
add_executable( sift src/sift.cpp )
target_link_libraries( sift ${OpenCV_LIBS} )
