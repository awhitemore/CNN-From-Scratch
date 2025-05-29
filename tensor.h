#ifndef TENSOR_H
#define TENSOR_H

#include <string>
#include <vector>

class Tensor{
    public:
        Tensor(int d,int h,int w);

        std::vector<float> flatten() const;

        float& at(int d, int h, int w);
        float at(int c, int r, int col) const;
        std::vector<std::vector<float>>& at(int filter_index);

        int depth() const { return depth_; }
        int height() const { return height_; }
        int width() const { return width_; }

        const std::vector<std::vector<std::vector<float>>>& data() const;
        void print() const;
    private:
        std::vector<std::vector<std::vector<float>>> data_;
        int depth_;
        int width_;
        int height_;
};

#endif