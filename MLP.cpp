#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

class Neuron {
private:
    vector<double> weights;
    double bias;
    double output;

public:
    Neuron(int numInputs) {
        
        for (int i = 0; i < numInputs; i++) {
            double weight = ((double)rand() / RAND_MAX) * 2 - 1;
            weights.push_back(weight);
        }

        bias = ((double)rand() / RAND_MAX) * 2 - 1;
    }

    double activate(vector<double>& inputs) {
        double sum = 0;

        for (int i = 0; i < inputs.size(); i++) {
            sum += inputs[i] * weights[i];
        }

        sum += bias;

        output = 1 / (1 + exp(-sum));

        return output;
    }

    double getOutput() {
        return output;
    }
};

class MLP {
private:
    vector<vector<Neuron>> layers;

public:
    MLP(vector<int> layerSizes) {
        int numLayers = layerSizes.size();

        for (int i = 0; i < numLayers; i++) {
            int numInputs = (i == 0) ? 1 : layerSizes[i - 1];
            int numNeurons = layerSizes[i];

            vector<Neuron> neuronLayer;

            for (int j = 0; j < numNeurons; j++) {
                Neuron neuron(numInputs);
                neuronLayer.push_back(neuron);
            }

            layers.push_back(neuronLayer);
        }
    }

    vector<double> feedForward(vector<double>& inputs) {
        vector<double> layerOutputs;

        vector<double> prevLayerOutputs = inputs;

        for (int i = 0; i < layers.size(); i++) {
            vector<double> currentLayerOutputs;

            for (int j = 0; j < layers[i].size(); j++) {
                double neuronOutput = layers[i][j].activate(prevLayerOutputs);
                currentLayerOutputs.push_back(neuronOutput);
            }

            layerOutputs = currentLayerOutputs;
            prevLayerOutputs = currentLayerOutputs;
        }

        return layerOutputs;
    }
};

int main() {
    srand(42);

    vector<int> layerSizes = { 5, 2, 1 };
    MLP network(layerSizes);

    vector<double> inputs = { 0.5, 0.3, 0.1 };

    vector<double> outputs = network.feedForward(inputs);

    cout << "Outputs: ";
    for (double output : outputs) {
        cout << output << " ";
    }
    cout << '\n';

    return 0;
}