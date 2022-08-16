//
// Created by Jakub Górski on 16/08/2022.
//

#include "../headers/NeuralNetwork.h"

Scalar activationFunction(Scalar x){
    return tanhf(x);
}

Scalar activationFunctionDerivative(Scalar x){
    return 1 - tanhf(x) * tanhf(x);
}

NeuralNetwork::NeuralNetwork(std::vector<unsigned int> topology, Scalar learningRate)
    : m_Topology(topology), learningRate(learningRate){
    for (unsigned int i = 0; i < m_Topology.size(); i++){
        if (i == m_Topology.size() - 1)
            neuronLayers.push_back(new RowVector(m_Topology[i]));
        else
            neuronLayers.push_back(new RowVector(m_Topology[i]+1));
        cacheLayers.push_back(new RowVector(neuronLayers.size()));
        deltas.push_back(new RowVector(neuronLayers.size()));

        if (i != topology.size()){
            neuronLayers.back()->coeffRef(m_Topology[i]) = 1.0;
            cacheLayers.back()->coeffRef(m_Topology[i]) = 1.0;
        }

        if (i > 0) {
            if (i != this->m_Topology.size() - 1) {
                weights.push_back(new Matrix(m_Topology[i - 1] + 1, m_Topology[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(m_Topology[i]).setZero();
                weights.back()->coeffRef(m_Topology[i - 1], m_Topology[i]) = 1;
            } else {
                weights.push_back(new Matrix(m_Topology[i - 1] + 1, m_Topology[i]));
                weights.back()->setRandom();
            }
        }
    }
}

void NeuralNetwork::propagateForward(RowVector &input) {
    neuronLayers.front()->block(0, 0, 1, neuronLayers.front()->size() - 1) = input;

    for (unsigned int i = 1; i < m_Topology.size(); i++){
        *neuronLayers[i] = *neuronLayers[i - 1] * *weights[i - 1];
        neuronLayers[i]->block(0, 0, 1, m_Topology[i]).unaryExpr(std::ptr_fun(activationFunction));
    }
}

void NeuralNetwork::propagateBackward(RowVector &output) {
    calcErrors(output);
    UpdateWeights();
}

void NeuralNetwork::calcErrors(RowVector &output) {
    *deltas.back() = output - *neuronLayers.back();
    for (unsigned int i = m_Topology.size() - 2; i > 0; i--){
        *deltas[i] = *deltas[i + 1] * weights[i]->transpose();
    }
}

void NeuralNetwork::UpdateWeights() {
    for (unsigned int i = 0; i < m_Topology.size() - 1; i++){
        if (i != m_Topology.size() - 2){
            for (unsigned int c = 0; c < weights[i]->cols() - 1; c++){
                for (unsigned int r = 0; r < weights[i]->rows(); r++){
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
        else
        {
            for (unsigned int c = 0; c < weights[i]->cols(); c++){
                for (unsigned int r = 0; r < weights[i]->rows(); r++){
                    weights[i]->coeffRef(r, c) += learningRate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(cacheLayers[i + 1]->coeffRef(c)) * neuronLayers[i]->coeffRef(r);
                }
            }
        }
    }
}

void NeuralNetwork::train(std::vector<RowVector *> input, std::vector<RowVector*> output) {
    for (unsigned int i = 0; i < input.size(); i++){
        std::cout << "Input to neural network: " << *input[i] << std::endl;
        propagateForward(*input[i]);
        std::cout << "Expected output: " << *output[i] << std::endl;
        std::cout << "Produced output: " << *neuronLayers.back() << std::endl;
        propagateBackward(*output[i]);
        std::cout << "MSE: " << std::sqrt((*deltas.back()).dot(*deltas.back()) / (float)deltas.back()->size()) << std::endl;
    }
}