#ifndef MLP_H
#define MLP_H

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>

class MLP {
    typedef boost::numeric::ublas::matrix<double> Mdouble;
    typedef vector<int> vec;
private:
    
    vector<ActivationFunction*>f; //Neuronas por capa
    
    vec pPerLayer; //Numero de perceptrones por capa (hidden layer)

    vector<Mdouble> pesos; //Pesos

    vector<Mdouble> outputs; //Outputs antes de la funcion de activacion
    
    int n_features,n_outputs,n_capas; // num capas ocultas
    
    vector <vector<double>> feature_vectors; //Vectores Caracteristicas
    vector <bool> Y;

    vector <vector<double>> X_train, X_test;
    vector<bool> Y_train, Y_test;
    
    Mdouble input;
    Mdouble MLP_ouput;

public: 
    MLP(vec initial_states,int n_out,vector<ActivationFunction*> actfunct) {
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
        ifstream archivo("data.csv");
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
                vector <double> filaData;
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