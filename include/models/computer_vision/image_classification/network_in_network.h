#pragma once

#include "../../common.h"

namespace xt::models {
    // The Network-in-Network model, built from scratch for MNIST.
    struct NetworkInNetwork : xt::Module {
        torch::nn::Sequential mlpconv1, mlpconv2, mlpconv3;

        torch::nn::AdaptiveAvgPool2d global_avg_pool;

        NetworkInNetwork(int num_classes = 10);
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);
    };



//    struct NetworkInNetwork : xt::Cloneable<NetworkInNetwork>
//    {
//    private:
//
//    public:
//        NetworkInNetwork(int num_classes /* classes */, int in_channels = 3/* input channels */);
//
//        NetworkInNetwork(int num_classes, int in_channels, std::vector<int64_t> input_shape);
//
//        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//
//        void reset() override;
//    };
}
