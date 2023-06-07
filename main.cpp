#include <bits/stdc++.h>
using namespace std;

class ActivationFunction {
    private:
        double value;
    public:
        double sigmoid(double value) {
            return 1 / (1 + exp(-value));
        }

        double relu(double value) {
            return max(0.0, value);
        }

        double tanh(double value) {
            return (exp(value) - exp(-value)) / (exp(value) + exp(-value));
        }

        double apply(string activationFunction, double value) {
            if (activationFunction == "sigmoid") {
                return sigmoid(value);
            } else if (activationFunction == "relu") {
                return relu(value);
            } else if (activationFunction == "tanh") {
                return tanh(value);
            } else {
                return value;
            }
        }
};

// class Neuron {
//     private:
//         double value;
//         double bias;
//         double weight;
//         ActivationFunction activationFunction;
//     public:
//         Neuron(double value, double bias, double weight, ActivationFunction activationFunction) {
//             this->value = value;
//             this->bias = bias;
//             this->weight = weight;
//             this->activationFunction = activationFunction;
//         }
// };

// class Layer {
//     private:
//         Neuron neurons[];
//     public:
//         Layer(Neuron neurons[]) {
//             this->neurons = neurons;
//         }
// };

// class MLP {
//     private:
//         Layer layers[];
//     public:
//         MLP(Layer layers[]) {
//             this->layers = layers;
//         }
// };

int main() {
    cout << "Hello World!" << endl;

    // Initalize MLP with data

    // Initialize layer and neurons

    // Initialize weights

    // Initialize bias

    // Initialize activation function

    ActivationFunction activationFunction;
    double value = activationFunction.apply("sigmoid", 0.5);
    // Print value double
    cout << value << endl;
    return 0;
}