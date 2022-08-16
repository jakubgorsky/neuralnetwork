cmake_minimum_required(VERSION 3.23)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 14)

set(HEADER_FILES)

set(SOURCE_FILES
        src/main.cpp)

add_executable(NeuralNetwork src/main.cpp)
