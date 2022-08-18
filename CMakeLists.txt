cmake_minimum_required(VERSION 3.23)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 14)

include_directories("C:\\libs\\eigen-3.4.0")

set(HEADER_FILES
        headers/NeuralNetwork.h headers/Neuron.h headers/TrainingData.h)

set(SOURCE_FILES
        src/NeuralNetwork.cpp
        src/main.cpp)

add_executable(NeuralNetwork ${SOURCE_FILES} ${HEADER_FILES})
