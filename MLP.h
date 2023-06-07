#ifndef MLP_H
#define MLP_H

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>

class MLP {
    typedef boost::numeric::ublas::matrix<double> Mdouble;
    typedef vector<int> Vi;
private:
    
    int tab[4] = {0,0,0,0}; //Matriz de confusion
    
    vector<ActivationFunction*>f; //Neuronas por capa
    
    Vi pPerLayer; //Numero de perceptrones por capa (hidden layer)

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
    MLP(Vi initial_states,int n_out,vector<ActivationFunction*> actfunct) {
        pPerLayer = initial_states;
        n_capas = pPerLayer.size();
        n_outputs = n_out;
        f = actfunct;
        load_data();
        n_features = feature_vectors[0].size();

        initialization();
        shuffleAndSplit();
    }

    void load_data(){}

    void initialization(){}

    void fit(int epochs,double lrate){}

    void forward(vector<double> X_f,int flag) {}

    void backward(int imgid,double lrate,int epoch) {}

    void shuffleAndSplit(float train=1) {}

    void testing(){}
    
    void init_matrix(Mdouble *mat){} //Inicializacion de matrices de pesos con valores random
};

#endif