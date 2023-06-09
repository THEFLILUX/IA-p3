#ifndef MLP_H
#define MLP_H

#include <boost/container/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <iostream>
#include <fstream>

typedef boost::numeric::ublas::matrix<double> Mdouble;
typedef boost::container::vector<int> vec;

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

class MLP {
private:

    boost::container::vector<ActivationFunction*> f; //Neuronas por capa

    vec pPerLayer; //Numero de perceptrones por capa (hidden layer)

    boost::container::vector<Mdouble> pesos; //Pesos

    boost::container::vector<Mdouble> outputs; //Outputs antes de la funcion de activacion

    int n_features,n_outputs,n_capas; // num capas ocultas

    boost::container::vector <boost::container::vector<double>> feature_vectors; //Vectores Caracteristicas
    boost::container::vector <bool> Y;

    boost::container::vector <boost::container::vector<double>> X_train, X_test;
    boost::container::vector<bool> Y_train, Y_test;

    Mdouble input;
    Mdouble MLP_ouput;

public:
    MLP(vec initial_states,int n_out,boost::container::vector<ActivationFunction*> actfunct) {
        pPerLayer = initial_states;
        n_capas = pPerLayer.size();
        n_outputs = n_out;
        f = actfunct;
        load_data();
        n_features = feature_vectors[0].size();

        initialization();
        shuffleAndSplit();
    }

    void load_data(){
        ifstream archivo("train.csv");
        if (archivo.is_open()) {
            string campos[128], fila;
            int numeroFilas = 0;
            while (!archivo.eof()) {
                getline(archivo, fila);
                istringstream stringStream(fila);
                unsigned int contador = 0;
                while (getline(stringStream, fila, ',')) {
                    campos[contador] = fila;
                    contador++;
                }
                boost::container::vector <double> filaData;
                for (int i = 0; i < 128; i++) {
                    if (i != 1) {
                        filaData.emplace_back(stof(campos[i]));
                    } else {
                        if (campos[i] == "M") {
                            Y.emplace_back(true);
                        } else {
                            Y.emplace_back(false);
                        }
                    }
                }
                feature_vectors.emplace_back(filaData);
                numeroFilas++;
            }
        }
        archivo.close();
    }

    void initialization(){}

    void init_matrix(Mdouble *mat){} //Inicializacion de matrices de pesos con valores random

    void fit(int epochs,double lrate){}

    void forward(vector<double> X_f,int flag) {}

    void backward(int imgid,double lrate,int epoch) {}

    void shuffleAndSplit(float train=1) {}

    void testing(){}
};

#endif

int main(){
    boost::container::vector<ActivationFunction*> actfunct;
    actfunct.push_back(new ActivationFunction());
    
    vec initial_states(2, 3);
    initial_states.pop_back(); 

    MLP mlp(initial_states, 1, actfunct);
    return 0;
}