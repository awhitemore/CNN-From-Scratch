#ifndef DENSE_H
#define DENSE_H

#include <vector>
#include "tensor.h"

class Dense{
    public:

        Dense(int in_size, int out_size);
    
        std::vector<float> forwardProp(const std::vector<float> flat);

        std::vector<float> backProp(const std::vector<float>& grad_output, float learning_rate);

    private:
        std::vector<std::vector<float>> weights;
        std::vector<float> biases;
        int input_size;
        int output_size;
        std::vector<float> input_vector;
};

#endif