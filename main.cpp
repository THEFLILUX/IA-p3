#include "MLP.h"
#include <fstream>
#include <deque>
#include <iomanip>
#include <sstream>

std::vector<float> normalizeVector(const std::vector<float>& values) {
    float minVal = *std::min_element(values.begin(), values.end());
    float maxVal = *std::max_element(values.begin(), values.end());
    
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

int main() {
    std::srand(21);

    std::ifstream file("train.csv");
    std::string line;
    std::getline(file, line);  
    
    lynalg::Matrix<float> data(1944, 128);
    lynalg::Matrix<float> labels(1944, 1);

    int cont = 0; 
    int cont2 = 0;

    while (std::getline(file, line) and cont < 1944) {
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

    //normalize data and labels
    labels.data = normalizeVector(labels.data); 

    data = data.add_scalar(1000);

    data.data = normalizeVector(data.data);

    // data.data=unnormalizeVector(data.data, , );
    
    int in_channels, out_channels, hidden_units_per_layer, hidden_layers;
    float lr;

    auto model = make_model(in_channels=128, out_channels=1, hidden_units_per_layer=200, hidden_layers=1, lr=.5f);

    // // train
    
    float mse;
    std::deque<float> deque;

    int epochs = 3;
    int min_loss = 1e9;
    
    for(int epoch = 0; epoch < epochs; ++epoch) {

        deque.clear();
        
        for(int i = 0; i < 1944; ++i) {
            // generate (x, y) training data
            auto x = data.get_row(i);
            auto y = labels.get_row(i);

            auto y_hat = model(x.transpose());  // forward pass
            model.backprop(y); // backward pass

            // compute and print error
            mse = (y - y_hat).square().data[0];
            deque.push_back(mse);
            
        }
    
        std::cout << std::setprecision(4) << "iter: " << epoch << " -- loss: " << mean(deque) << std::endl;
        
    }

    std::ifstream file2("test.csv");
    std::getline(file2, line);  
    lynalg::Matrix<float> data2(487, 128);
    lynalg::Matrix<float> labels2(487, 1);
    cont = 0; 
    cont2 = 0;

    while (std::getline(file2, line) and cont < 487) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<float> rowData;

        while (std::getline(lineStream, cell, ',')) {
            rowData.push_back(std::stod(cell));
        }
        
        // Label is the last element of the row
        labels2.data[cont] = rowData.back();

        // Data are all elements except the last one
        for (int i = 0; i < rowData.size()-1; i++) {
            data2.data[cont2] = rowData[i];
            cont2++;
        }

        cont++;
    }

    labels2.data = normalizeVector(labels2.data); 
    data2 = data2.add_scalar(1000);
    data2.data = normalizeVector(data2.data);

    float cnt = 0;
    labels2.data = unnormalizeVector(labels2.data, 1, 24);

    int truePositives = 0;
    int falsePositives = 0;
    int falseNegatives = 0;

    for (int i = 0; i < labels2.data.size(); i++) {
    auto x = data2.get_row(i);
    auto y = labels2.get_row(i);
    auto y_hat = model(x.transpose());
    y_hat.data = unnormalizeVector(y_hat.data, 1, 24);

    if (y.data[0] - y_hat.data[0] < 1) {
        truePositives++;
    } else {
        falseNegatives++;
    }
    
    if (y_hat.data[0] < y.data[0] - 1 || y_hat.data[0] > y.data[0] + 1) {
        falsePositives++;
    }
    }

    float accuracy = static_cast<float>(truePositives) / labels2.data.size();
    std::cout << "Accuracy: " << accuracy << std::endl;

    float precision = static_cast<float>(truePositives) / (truePositives + falsePositives);
    std::cout << "Precision: " << precision << std::endl;

    float recall = static_cast<float>(truePositives) / (truePositives + falseNegatives);
    std::cout << "Recall: " << recall << std::endl;

    float f1Score = 2.0 * (precision * recall) / (precision + recall);
    std::cout << "F1 Score: " << f1Score << std::endl;

    return 0;
}