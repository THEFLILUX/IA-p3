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
            } 
            else if (activationFunction == "relu") {
                return relu(value);
            }
            else if (activationFunction == "tanh") {
                return tanh(value);
            } 
            else {
                return value;
            }
        }
};

int main() {
    cout << "Hello World!" << endl;

    ActivationFunction activationFunction;
    double value = activationFunction.apply("sigmoid", 0.5);

    cout << value << endl;
    return 0;
}