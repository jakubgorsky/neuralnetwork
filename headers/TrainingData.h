//
// Created by Jakub Górski on 18/08/2022.
//

#ifndef NEURALNETWORK_TRAININGDATA_H
#define NEURALNETWORK_TRAININGDATA_H
#include <sstream>
#include <fstream>
#include <vector>
class TrainingData {
public:
    explicit TrainingData(const std::string& filename){
        m_trainingDataFile.open(filename.c_str());
    }
    bool isEof() { return m_trainingDataFile.eof(); }

    void getTopology(std::vector<unsigned> &topology){
        std::string line, label;
        getline(m_trainingDataFile, line);
        std::stringstream ss(line);
        ss >> label;
        if(this->isEof() || label != "topology:")
            abort();
        while(!ss.eof()){
            unsigned n;
            ss >> n;
            topology.push_back(n);
        }
    }

    unsigned getNextInputs(std::vector<double> &inputVals){
        inputVals.clear();
        std::string line;
        getline(m_trainingDataFile, line);
        std::stringstream ss(line);
        std::string label;
        ss >> label;
        if(label == "in:"){
            double oneValue;
            while(ss >> oneValue){
                inputVals.push_back(oneValue);
            }
        }

        return inputVals.size();
    }

    unsigned getTargetOutputs(std::vector<double> &targetOutputVals){
        targetOutputVals.clear();

        std::string line, label;
        getline(m_trainingDataFile, line);
        std::stringstream ss(line);
        ss >> label;
        if (label == "out:"){
            double oneValue;
            while(ss >> oneValue){
                targetOutputVals.push_back(oneValue);
            }
        }
        return targetOutputVals.size();
    }
private:
    std::ifstream m_trainingDataFile;
};
#endif //NEURALNETWORK_TRAININGDATA_H
