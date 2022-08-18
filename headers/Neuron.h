//
// Created by Jakub Górski on 17/08/2022.
//

#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include <cstdlib>
#include <cmath>

class Neuron;
typedef std::vector<Neuron> Layer;

struct Connection {
    Connection() { weight = randomWeight();}
    double weight{};
    double deltaWeight{};
private:
    static double randomWeight() { return rand() / double(RAND_MAX); }
};

template <typename T>
T max(T x, T y){
    return (x > y) ? x : y;
}

class Neuron {
public:
    Neuron(unsigned int numOutputs, unsigned index)
        : m_Index(index){
        for (unsigned c = 0; c < numOutputs; ++c){
            m_outputWeights.emplace_back();
        }
        eta = 0.2;
        alpha = 0.6;
    }

    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal() const { return m_outputVal; }

    void feedForward(Layer &prevLayer){
        double sum = 0.0;

        for (unsigned n = 0; n < prevLayer.size(); ++n){
            sum += prevLayer[n].getOutputVal() * prevLayer[n].m_outputWeights[m_Index].weight;
        }

        m_outputVal = Neuron::transferFunction(sum);
    }

    void calcOutputGradients(double targetVal){
        double delta = targetVal - m_outputVal;
        m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
    }

    void calcHiddenGradients(const Layer &nextLayer){
        double dow = sumDOW(nextLayer);
        m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
    }

    void updateInputWeights(Layer &prevLayer){
        for (auto & neuron : prevLayer){
            double oldDeltaWeight = neuron.m_outputWeights[m_Index].deltaWeight;

            double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient
                    + alpha * oldDeltaWeight;

            neuron.m_outputWeights[m_Index].deltaWeight = newDeltaWeight;
            neuron.m_outputWeights[m_Index].weight += newDeltaWeight;
        }
    }

    double eta;
    double alpha;

private:
    static double transferFunction(double x) {
        return tanh(x);
    }
    static double transferFunctionDerivative(double x) {
        return 1 - x * x;
    }
    double sumDOW(const Layer &nextLayer) const{
        double sum = 0.0;
        for (unsigned n = 0; n < nextLayer.size() - 1; ++n){
            sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
        }
        return sum;
    }
    unsigned m_Index{};
    double m_outputVal{};
    double m_gradient{};
    std::vector<Connection> m_outputWeights;
};


#endif //NEURALNETWORK_NEURON_H
