#ifndef MLP_H
#define MLP_H

#include <boost/container/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/random.hpp>
#include "ActivationFunction.cpp"
using namespace std;

class MLP {
    typedef boost::numeric::ublas::matrix<double> Mdouble;
    typedef boost::container::vector<int> vec;
private:

    boost::container::vector<ActivationFunction*>f; //Neuronas por capa

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

    void initialization(){
        Mdouble aux;
        input = Mdouble(1,n_features);
        init_matrix(&input);

        //output = Mdouble(1,n_outputs);
        //init_matrix(&output);

        for(int i = 0;i<n_capas;i++){
            //Creo matriz de pesos y lo agrego al vector de pesos
            if(i==0){
                aux = Mdouble(n_features,pPerLayer[i]);
                init_matrix(&aux);
                pesos.push_back(aux);
            }
            else {
                aux = Mdouble(pPerLayer[i-1],pPerLayer[i]);
                init_matrix(&aux);
                pesos.push_back(aux);
            }
        }
        aux = Mdouble(pPerLayer[n_capas-1],n_outputs);
        init_matrix(&aux);
        pesos.push_back(aux);
    }

    void init_matrix(Mdouble *mat){ //Inicializacion de matrices de pesos con valores random
        std::time_t now = std::time(0);
        boost::random::mt19937 gen{static_cast<std::uint32_t>(now)};
        boost::random::uniform_int_distribution<> dist{1, 10000};
        for (int i = 0;i< mat->size1();i++){
            for (int j = 0;j< mat->size2();j++)
                mat->operator()(i,j)= double(dist(gen)/10000.0);
        }
    }

    void fit(int epochs,double lrate){
        for(int i = 0;i<epochs;i++){
            cout<<"Epoch"<<i+1<<endl;
            for(int j = 0;j<X_train.size();j++){
                //cout<<"size"<<X_f.size()<<endl;
                forward(X_train[j],0);
                backward(j,lrate,i);
            }
        }
    }

    void forward(boost::container::vector <double> X_f,int flag) {}

    void backward(int imgid,double lrate,int epoch) {}

    void shuffleAndSplit(float train=1) {
        int ft_size = feature_vectors.size();
        // Realizar el shuffle de la data
        srand(10);
        for (int i=0; i<ft_size; i++) {
            int j = i + rand() % (ft_size - i);
            swap(feature_vectors[i], feature_vectors[j]);
            swap(Y[i], Y[j]);
        }
        for (int i=0; i<ft_size; i++) {
            if (i < ft_size*train) {
                X_train.push_back(feature_vectors[i]);
                Y_train.push_back(Y[i]);
            } else {
                X_test.push_back(feature_vectors[i]);
                Y_test.push_back(Y[i]);
            }
        }
    }
};

#endif