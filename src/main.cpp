#include <cassert>
#include "../headers/TrainingData.h"
#include "../headers/NeuralNetwork.h"
#include <chrono>


void showVectorVals(std::string label, std::vector<double> &v){
    std::cout << label << " ";
    for (auto i : v){
        std::cout << i << " ";
    }
    std::cout << "\n";
}

int main() {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;
    std::ios_base::sync_with_stdio(false);


    TrainingData trainingData("data.txt");

    std::vector<unsigned int> topology;
    trainingData.getTopology(topology);
    Net net(topology);

    std::vector<double> inputVals, targetVals, resultVals;
    int pass = 0;

    auto start = high_resolution_clock::now();

    while (!trainingData.isEof()){
        ++pass;
        std::cout << "\nPass: " << pass;
        if (trainingData.getNextInputs(inputVals) != topology[0])
            break;
        showVectorVals(": Inputs:", inputVals);
        net.feedForward(inputVals);
        net.getResults(resultVals);
        showVectorVals("Outputs:", resultVals);

        trainingData.getTargetOutputs(targetVals);
        showVectorVals("Targets:", targetVals);
        assert(targetVals.size() == topology.back());

        net.backProp(targetVals);
    }
    auto end = high_resolution_clock::now();
    auto deltaT = duration_cast<milliseconds>(end - start);
    std::cout << "\n Training done.\n"
        << "Training time: " << deltaT.count() << "ms\n";
    return 0;
}
