#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <vector>
#include "tensor.h"

class Conv_layer{
    public:

        Conv_layer(int size, int stride);

        //applies filter to image and returns [1][26[26] tensor
        Tensor applyFilters(const Tensor& t);

        //returns kernel for each filter matrix from filters
        const std::vector<std::vector<float>> getFilter(int n) const;

        //computes dot product of image and kernel
        float dotProd(const std::vector<std::vector<float>>& k, const std::vector<std::vector<float>>& t);

        float Relu(const float& result);

        Tensor MaxPool(Tensor& input);

        Tensor MaxPoolBackProp(const std::vector<float>& gradients);

        std::vector<Tensor> getMax_Pool_Mask();

        void BackProp(Tensor &original, Tensor &d_out, float learning_rate);

    private:
        std::vector<Tensor> max_pool_mask; // Holds mask per filter per sample
        Tensor filters;
        int size, stride;
};

#endif