//
// Created by Jakub Górski on 16/08/2022.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <Eigen/Eigen>
#include <iostream>
#include <vector>

typedef float Scalar;
typedef Eigen::MatrixXf Matrix;
typedef Eigen::RowVectorXf RowVector;
typedef Eigen::VectorXf ColVector;

class NeuralNetwork {
public:
    NeuralNetwork(std::vector<unsigned int> topology, Scalar learningRate = Scalar(0.005));

    void propagateForward(RowVector& input);
    void propagateBackward(RowVector& output);
    void calcErrors(RowVector& output);
    void UpdateWeights();
    void train(std::vector<RowVector*> input, std::vector<RowVector*> output);

    std::vector<unsigned int> m_Topology;
    std::vector<RowVector*> neuronLayers;
    std::vector<RowVector*> cacheLayers;
    std::vector<RowVector*> deltas;
    std::vector<Matrix*> weights;
    Scalar learningRate;
};

#endif //NEURALNETWORK_NEURALNETWORK_H
