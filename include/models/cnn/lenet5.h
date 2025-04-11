#pragma once
#include <torch/torch.h>
#include "../base.h"
#include <iostream>
#include <vector>


using namespace std;

namespace xt::models {
    struct LeNet5 : BaseModel {
    protected:
        mutable  torch::nn::Sequential layer1 = nullptr, layer2 = nullptr;
        mutable  torch::nn::Linear fc1 = nullptr, fc2 = nullptr, fc3 = nullptr;

    public:
        LeNet5(int num_classes/* classes */, int in_channels = 1/*  input channels */);
        LeNet5(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        torch::Tensor forward(torch::Tensor x) const override;
    };
}
