#include "tensor.h"     
#include <iostream> 
#include <iomanip>

using namespace std;


Tensor::Tensor(int d, int h, int w)
    : depth_(d), width_(w), height_(h) {

    data_.resize(depth_);
    
    for (int i = 0; i < depth_; i++) {
        data_[i].resize(height_);
        for (int j = 0; j < height_; j++) {
            data_[i][j].resize(width_, 0.0f);
        }
    }
}

void Tensor::print() const{
    for (int d = 0; d < depth_; d++) {
        for (int h = 0; h < height_; h++) {
            for(int w = 0; w < width_; w++){
                cout << setprecision(0)<< data_[d][h][w] << " ";
            }
            cout << "\n";
        }
        cout << "Depth done\n";
    } 
}

vector<float> Tensor::flatten() const{
    vector<float> single;
    single.reserve(depth_ * height_ * width_);

    for (int d = 0; d < depth_; d++) {
        for (int h = 0; h < height_; h++) {
            for(int w = 0; w < width_; w++){
                single.push_back(data_[d][h][w]);
            }
        }
    }
    return single;
}

const std::vector<std::vector<std::vector<float>>>& Tensor::data() const{
    return data_;
}

float& Tensor::at(int d, int h, int w){
    return data_[d][h][w];
}

float Tensor::at(int c, int r, int col) const{
    return data_[c][r][col];
}
    
vector<vector<float>>& Tensor::at(int filter_index) {
    return data_.at(filter_index);
}
