cmake_minimum_required(VERSION 4.1)
project(MNIST_C__)

set(CMAKE_CXX_STANDARD 26)

if(MSVC)
    add_compile_options(/EHsc)
endif()


# Add an include directory
include_directories(include)

# Add your source files
set(SOURCES
    src/main.cpp
    src/dataset.cpp
    src/network.cpp
        include/matrix.h
        src/matrix.cpp
        include/DataLoaders/MNistLoader.h
        src/DataLoaders/MNistLoader.cpp
)

add_executable(MNIST_C__ ${SOURCES})

add_subdirectory(test)