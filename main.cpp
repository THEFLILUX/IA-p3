#include "MLP.h"
#include <fstream>
#include <deque>
#include <iomanip>

// helper to initialize multi-layer perceptron with n hidden layers each w/ same num hidden units
auto make_model(size_t in_channels, size_t out_channels, size_t hidden_units_per_layer, int hidden_layers, float lr) {
  std::vector<size_t> units_per_layer;

  units_per_layer.push_back(in_channels);

  for (int i = 0; i < hidden_layers; ++i)
    units_per_layer.push_back(hidden_units_per_layer);

  units_per_layer.push_back(out_channels);

  mlp::MLP<float> model(units_per_layer, 0.01f);
  return model;
}

auto mean(const auto &d) {
  float mu {0.};
  for(auto v: d){
    mu += v;
  }
  return mu/d.size();
}

void log(auto &file, const auto &x, const auto &y, const auto &y_hat){
  auto mse = (y.data[0] - y_hat.data[0]);
  mse = mse*mse;

  file << mse << " "
       << x.data[0] << " "
       << y.data[0] << " "
       << y_hat.data[0] << " \n";
}

int main() {
    std::srand(21);

    std::ifstream file("train.csv");
    std::string line;
    std::getline(file, line);  
    
    lynalg::Matrix<float> data(487, 128);
    lynalg::Matrix<float> labels(487, 1);

    int cont = 0; 
    int cont2 = 0;

    while (std::getline(file, line) and cont < 487) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> rowData;

        while (std::getline(lineStream, cell, ',')) {
            rowData.push_back(std::stod(cell));
        }
        
        // Label is the last element of the row
        labels.data[cont] = rowData.back();

        // Data are all elements except the last one
        for (int i = 0; i < rowData.size()-1; i++) {
            data.data[cont2] = rowData[i];
            cont2++;
        }

        cont++;
    }

    data.print_shape();
    labels.print_shape();
   
    int in_channels, out_channels, hidden_units_per_layer, hidden_layers;
    float lr;

    auto model = make_model(in_channels=128, out_channels=1, hidden_units_per_layer=50, hidden_layers=1, lr=.5f);

    // // train
    std::ofstream my_file;
    my_file.open ("train.text");
    int print_every{10};

    float mse;
    auto deque = std::deque<float>(print_every);

    data.get_row(0).print_shape();

    for(int i = 0; i < 487; ++i) {
        auto x = data.get_row(i);
        auto y = labels.get_row(i);

        auto y_hat = model(x.transpose());  // forward pass
        model.backprop(y); // backward pass

        // compute and print error
        mse = (y - y_hat).square().data[0];
        deque.push_back(mse);

        if ((i+1)%print_every==0) {
            log(my_file, x, y, y_hat);
            my_file << mse << " " << x.data[0] << " " << y.data[0] << " " << y_hat.data[0] << " \n";
            std::cout << std::setprecision(4) << std::scientific << "iter: " << i << " -- loss: " << mean(deque) << std::endl;
        }
    }

    //print accuracy:
    int cnt = 0;
    for (int i = 0; i<labels.data.size(); i++){
        auto x = data.get_row(i);
        auto y = labels.get_row(i);
        auto y_hat = model(x.transpose());
        
        if (round(y.data[i]) == round(y_hat.data[i])){
            cnt++;
        }
    }
    
    float acc = cnt/labels.data.size();
    std::cout << "Accuracy: " << acc << std::endl;

    my_file.close();

    return 0;
}