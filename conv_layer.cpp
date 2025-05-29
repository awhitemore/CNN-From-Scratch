#include "conv_layer.h"
#include "tensor.h"     
#include <iostream> 

using namespace std;

Conv_layer::Conv_layer(int sz, int st)
    :filters(8, 3, 3),size(sz), stride(st){
    
    vector<vector<float>> filter1 = {
        {1, 0, -1},
        {1, 0, -1},
        {1, 0, -1}
    };
    vector<vector<float>> filter2 = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };
    vector<vector<float>> filter3 = {
        {1, 1, 1},
        {0, 0, 0},
        {-1, -1, -1}
    };
    std::vector<std::vector<float>> filter4 = {
        {-1, -1, -1},
        {-1,  8, -1},
        {-1, -1, -1}
    };
    
    std::vector<std::vector<float>> filter5 = {
        {0, -1, 0},
        {-1, 5, -1},
        {0, -1, 0}
    };
    
    std::vector<std::vector<float>> filter6 = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}
    };
    
    std::vector<std::vector<float>> filter7 = {
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0}
    };
    std::vector<std::vector<float>> filter8 = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    filters.at(0) = filter1;
    filters.at(1) = filter2;
    filters.at(2) = filter3;
    filters.at(3) = filter4;
    filters.at(4) = filter5;
    filters.at(5) = filter6;
    filters.at(6) = filter7;
    filters.at(7) = filter8;       
}

Tensor Conv_layer::applyFilters(const Tensor& t){
    int num_layers = size;
    int o_height = (t.height() - 3) / stride + 1;
    int o_width = (t.width() - 3) / stride + 1;
    Tensor output(num_layers,o_height,o_width);


    for(int f = 0; f < size; f++) { //filter loop
        vector<vector<float>> filter = filters.at(f);
        for (int h = 0; h < o_height; h++) {
            for (int w = 0; w < o_width; w++) {
                vector<vector<float>> patch(3, vector<float>(3, 0.0f));
                for(int i = 0; i < 3;i++){                        
                    for(int j = 0; j < 3;j++){
                        patch[i][j] = t.at(0,h*stride + i,stride*w + i);
                    }
                }
                output.at(f,h,w) = Relu(dotProd(filter,patch));
            }
        } 
    }
    return output;
}

float Conv_layer::dotProd(const std::vector<std::vector<float>>& k, const std::vector<std::vector<float>>& t){
    float sum = 0;
    for(int i = 0;i < 3;i++){
        for(int j = 0;j < 3; j++){
            sum += k[i][j] * t[i][j];
        }
    }
    return sum;
}

float Conv_layer::Relu(const float& result){
    return (result > 0) ? result : 0;
}

Tensor Conv_layer::MaxPool(Tensor& input){ 
    int o_height = input.height()/2;
    int o_width = input.width()/2;
    Tensor output(size, o_height, o_width); //8, 13, 13
    Tensor mask(size, input.height(), input.width()); //8, 26, 26

    for(int f = 0; f < size; f++) { //layer loop, aka activation layers
        for (int h = 0; h < o_height; h++) {
            for (int w = 0; w < o_width; w++) {
                float max_val = -INFINITY;  // Use float and start with very small value
                int max_r = -1, max_c = -1;
                
                for(int i = 0; i < 2; i++){
                    for(int j = 0; j < 2; j++){
                        float p = input.at(f, i+2*h, j+2*w);  // Use f, not 0
                        if(p > max_val) {
                            max_val = p;
                            max_r = i+2*h;
                            max_c = j+2*w;
                        }
                    }
                }
                mask.at(f, max_r, max_c) = 1.0f;
                output.at(f, h, w) = max_val;
            } 
        }
    }

    if (max_pool_mask.empty()) {
        max_pool_mask.push_back(mask);
    } else {
        max_pool_mask[0] = mask; // Overwrite the previous mask
    }
    return output;
}

Tensor Conv_layer::MaxPoolBackProp(const vector<float>& gradients) {
    // Use the mask for the specific sample  
    Tensor grad_output = max_pool_mask[0];

    int loc = 0;
    
    // Iterate through each pooled position
    for (int f = 0; f < size; f++) { // size should be number of filters
        for (int h = 0; h < grad_output.height() / 2; h++) {
            for (int w = 0; w < grad_output.width() / 2; w++) {
                // Find the position of the max value in this 2x2 pool
                for (int i = 0; i < 2; i++) {
                    for (int j = 0; j < 2; j++) {
                        int y_pos = i + 2 * h;
                        int x_pos = j + 2 * w;
                        
                        // Check if this position was the max in the original pooling
                        if (grad_output.at(f, y_pos, x_pos) == 1.0f) {
                            grad_output.at(f, y_pos, x_pos) = gradients[loc];
                            break; // Found the max position for this pool
                        }
                    }
                }
                loc++;
            }
        }
    }
    return grad_output;
}

vector<Tensor> Conv_layer::getMax_Pool_Mask(){
    return max_pool_mask;
}

void Conv_layer::BackProp(Tensor &original, Tensor &d_out, float learning_rate){
    vector<vector<vector<float>>> grad_filters(size,vector<vector<float>>(3, vector<float>(3, 0.0f)));
    // Compute gradients w.r.t. filter weights
    for (int f = 0; f < size; ++f) {
        for (int h = 0; h < d_out.height(); ++h) {
            for (int w = 0; w < d_out.width(); ++w) {
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int in_h = h + kh;
                        int in_w = w + kw;
                        grad_filters[f][kh][kw] += original.at(0, in_h, in_w) * d_out.at(f, h, w);
                    }
                }
            }
        }
    }

    // Update filter weights
    for (int f = 0; f < size; ++f) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                filters.at(f)[kh][kw] -= learning_rate * grad_filters[f][kh][kw];
            }
        }
    }
}