#pragma once

#include "include/models/common.h"


using namespace std;

namespace xt::models
{
    struct LeNet5 : xt::Cloneable<LeNet5>
    {
    protected:
        mutable torch::nn::Sequential layer1 = {nullptr}, layer2 = {nullptr};
        mutable torch::nn::Linear fc1 = nullptr, fc2 = nullptr, fc3 = nullptr;

    public:
        explicit LeNet5(int num_classes/* classes */, int in_channels = 1/*  input channels */);
        LeNet5(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        // torch::Tensor forward(torch::Tensor x) const override;
        void reset() override;
    };
}
