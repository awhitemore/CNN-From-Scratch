#include "tensor.h"  
#include "dense.h"   
#include <iostream> 

using namespace std;

Dense::Dense(int in_size, int out_size) : input_size(in_size), output_size(out_size) {

    biases.resize(out_size, 0.0f);
    
    float weight_scale = sqrt(2.0f / input_size); 
        
    weights.resize(output_size, std::vector<float>(input_size));
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < input_size; j++) {
            // Random number between -weight_scale and weight_scale
            weights[i][j] = ((float)rand() / RAND_MAX * 2 - 1) * weight_scale;
        }
    }
}

vector<float> Dense::forwardProp(const vector<float> flat){
    this->input_vector = flat;
    vector<float> logits(10, 0.0f);
    for(int i = 0; i < 10;i++){
        logits[i] = biases[i];
        for(int j = 0;j < flat.size();j++){
            logits[i] += weights[i][j]*flat[j];
        }
    }
    return logits;
}

vector<float> Dense::backProp(const vector<float>& grad_output, float learning_rate) {
    int out_size = grad_output.size();            // Number of output neurons
    int in_size = input_vector.size();            // Number of input features
    
    vector<float> dl_dx(in_size, 0.0f);     // Gradient w.r.t. input vector

    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++) {
            dl_dx[j] += grad_output[i] * weights[i][j];  // Use original weights
        }
    }

    for (int i = 0; i < out_size; i++) {
        for (int j = 0; j < in_size; j++) {
            float grad_weight = grad_output[i] * input_vector[j];
            
            grad_weight = clamp(grad_weight, -100.0f, 100.0f);
            
            weights[i][j] -= learning_rate * grad_weight;
            if (!isfinite(weights[i][j])) weights[i][j] = 0.0f;
            weights[i][j] = clamp(weights[i][j], -10.0f, 10.0f);
        }
        
        biases[i] -= learning_rate * grad_output[i];
        if (!isfinite(biases[i])) biases[i] = 0.0f;
    }
    
    return dl_dx;
}
