#include "MLP.h"
#include <fstream>
#include <deque>
#include <iomanip>

std::vector<float> normalizeVector(const std::vector<float>& values) {
    // Find the minimum and maximum values in the vector
    float minVal = *std::min_element(values.begin(), values.end());
    float maxVal = *std::max_element(values.begin(), values.end());
    std::cout<<minVal<<" "<<maxVal<<std::endl;

    // Perform Min-Max normalization
    std::vector<float> normalizedValues;
    for (const auto& value : values) {
        float normalizedValue = (value - minVal) / (maxVal - minVal);
        normalizedValues.push_back(normalizedValue);
    }

    return normalizedValues;
}

// Function to unnormalize a vector
std::vector<float> unnormalizeVector(const std::vector<float>& normalizedValues, float minVal, float maxVal) {
    // Perform the inverse transformation
    std::vector<float> unnormalizedValues;
    for (const auto& value : normalizedValues) {
        float unnormalizedValue = value * (maxVal - minVal) + minVal;
        unnormalizedValues.push_back(unnormalizedValue);
    }

    return unnormalizedValues;
}

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

    //normalize data and labels
    labels.data = normalizeVector(labels.data); 

    data.data = normalizeVector(data.data);
   
    int in_channels, out_channels, hidden_units_per_layer, hidden_layers;
    float lr;

    auto model = make_model(in_channels=128, out_channels=1, hidden_units_per_layer=100, hidden_layers=3, lr=.5f);

    // // train
    std::ofstream my_file;
    my_file.open ("train.text");
    int print_every{401};

    float mse;
    auto deque = std::deque<float>(print_every);

    int epochs = 1000;
    int min_loss = 1e9;

    for(int epoch = 0; epoch < epochs; ++epoch) {

        deque.clear();
        
        for(int i = 0; i < 487; ++i) {
            // generate (x, y) training data
            auto x = data.get_row(i);
            auto y = labels.get_row(i);

            auto y_hat = model(x.transpose());  // forward pass
            model.backprop(y); // backward pass

            // compute and print error
            mse = (y - y_hat).square().data[0];
            deque.push_back(mse);
            
        }
    
        std::cout << std::setprecision(4) << std::scientific << "iter: " << epoch << " -- loss: " << mean(deque) << std::endl;
        
    }

    // print accuracy:
    float cnt = 0;
    labels.data = unnormalizeVector(labels.data, 1, 24);

    for (int i = 0; i<labels.data.size(); i++){
        auto x = data.get_row(i);
        auto y = labels.get_row(i);
        auto y_hat = model(x.transpose());
        y_hat.data = unnormalizeVector(y_hat.data, 1, 24);
        
        if (y.data[i] - y_hat.data[i] < 1){
            cnt++;
        }

        std::cout<< "y: " << y.data[i] << "\t||\t y_hat: " << y_hat.data[i] << std::endl;
    }
    
    float acc = cnt/487.f;
    std::cout << "Accuracy: " << acc << std::endl;

    my_file.close();

    return 0;
}