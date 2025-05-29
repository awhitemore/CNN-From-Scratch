#include <iostream>
#include "tensor.h"
#include "conv_layer.h"
#include "dense.h"
#include "softmax.h"

#include <cstdlib>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <string>

using namespace std;

uint32_t read_uint32(ifstream &f){
    uint8_t bytes[4];
    f.read((char*)bytes, 4);
    return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
}

vector<vector<uint8_t>> load_images(const string &filename, int &num_imgs){
    ifstream file(filename, ios::binary);
    if(!file.is_open()) throw runtime_error("Cannot open file: " + filename);

    uint32_t magic = read_uint32(file);
    if (magic != 2051) throw runtime_error("Invalid magic number in image file");

    num_imgs = read_uint32(file);
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);

    vector<vector<uint8_t>> images(num_imgs, vector<uint8_t>(rows * cols));
    for (int i = 0; i < num_imgs; ++i) {
        file.read((char*)images[i].data(), rows * cols);
    }
    return images;
}

vector<uint8_t> load_labels(const string &filename, int &num_labels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) throw runtime_error("Cannot open file: " + filename);

    uint32_t magic = read_uint32(file);
    if (magic != 2049) throw runtime_error("Invalid magic number in label file");

    num_labels = read_uint32(file);
    vector<uint8_t> labels(num_labels);
    file.read((char*)labels.data(), num_labels);
    return labels;
}

int main() {
    // Set random seed for reproducible results
    srand(42);
    vector<string> class_names = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    // Adjustable number of epochs
    const int NUM_EPOCHS = 3;

    // Load data
    int num_train, num_labels;
    auto train_images = load_images("train-images-idx3-ubyte", num_train);
    auto train_labels = load_labels("train-labels-idx1-ubyte", num_labels);

    int num_test, num_labels_test;
    auto test_images = load_images("t10k-images-idx3-ubyte", num_test);
    auto test_labels = load_labels("t10k-labels-idx1-ubyte", num_labels_test);

    float learning_rate = 0.001;

    Conv_layer convolution_layer(8, 1);        // 8 filters, 1 input channel
    Dense dense_layer(1352, 10);               // 1352 from flattened maxpool, 10 outputs

    // === Training Loop for Multiple Epochs ===
    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        cout << "\n=== Epoch " << (epoch + 1) << "/" << NUM_EPOCHS << " ===" << endl;
        
        float train_correct = 0.0f;
        float total_loss = 0.0f;
        
        for (int i = 0; i < num_train; ++i) {
            // --- Prepare one image as a tensor ---
            Tensor image_tensor(1, 28, 28);  // One image, one channel
            for (int r = 0; r < 28; ++r) {
                for (int c = 0; c < 28; ++c) {
                    image_tensor.at(0, r, c) = train_images[i][r * 28 + c] / 255.0f;
                }
            }

            // --- Forward Pass ---
            Tensor conv_out = convolution_layer.applyFilters(image_tensor);
            Tensor pooled = convolution_layer.MaxPool(conv_out);

            vector<float> flat = pooled.flatten();
            vector<float> dense_out = dense_layer.forwardProp(flat);

            vector<float> probs = softmax(dense_out);

            // --- Compute loss gradient ---
            int true_label = train_labels[i];
            vector<float> dL_dz = probs;
            dL_dz[true_label] -= 1.0f;

            // --- Backward Pass ---
            vector<float> gradients = dense_layer.backProp(dL_dz, learning_rate);

            Tensor d_maxpool = convolution_layer.MaxPoolBackProp(gradients);
            convolution_layer.BackProp(image_tensor, d_maxpool, learning_rate);

            int pred = distance(probs.begin(), max_element(probs.begin(), probs.end()));

            if(pred == true_label) train_correct++;

            // Calculate loss for tracking
            float loss = -log(std::max(probs[true_label], 1e-8f));
            total_loss += loss;

            if (i % 10000 == 0) {
                cout << "  Sample " << i << ", Loss: " << loss << endl;
                cout << "  Predicted: " << class_names[pred] << ", Actual: " << class_names[true_label] << endl;
            }
        }
        
        float epoch_accuracy = train_correct / num_train;
        float avg_loss = total_loss / num_train;
        
        cout << "Epoch " << (epoch + 1) << " - Train Accuracy: " << epoch_accuracy 
             << ", Average Loss: " << avg_loss << endl;
    }

    // === Testing Phase ===
    cout << "\n=== Testing Phase ===" << endl;
    float test_correct = 0.0f;
    
    for (int i = 0; i < num_test; ++i) {
        // --- Prepare one image as a tensor ---
        Tensor image_tensor(1, 28, 28);  // One image, one channel
        for (int r = 0; r < 28; ++r) {
            for (int c = 0; c < 28; ++c) {
                image_tensor.at(0, r, c) = test_images[i][r * 28 + c] / 255.0f;
            }
        }

        Tensor conv_out = convolution_layer.applyFilters(image_tensor);
        Tensor pooled = convolution_layer.MaxPool(conv_out);

        vector<float> flat = pooled.flatten();
        vector<float> dense_out = dense_layer.forwardProp(flat);

        vector<float> probs = softmax(dense_out);

        int true_label = test_labels[i];
        int pred = distance(probs.begin(), max_element(probs.begin(), probs.end()));

        if(pred == true_label) test_correct++;
    }
    
    cout << "Final Test Accuracy: " << test_correct / num_test << endl;

    return 0;
}