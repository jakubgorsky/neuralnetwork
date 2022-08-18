//
// Created by Jakub Górski on 16/08/2022.
//

#ifndef NEURALNETWORK_NEURALNETWORK_H
#define NEURALNETWORK_NEURALNETWORK_H

#include <iostream>
#include <vector>
#include <fstream>
#include "../headers/Neuron.h"

typedef std::vector<Neuron> Layer;

class Net {
public:
    Net(const std::vector<unsigned int> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backProp(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageError() const { return m_recentAverageError; }
private:
    std::vector<Layer> m_layers;
    double m_error{}, m_recentAverageError{};
};

#endif //NEURALNETWORK_NEURALNETWORK_H
