#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor star_relu(const torch::Tensor& x, double scale = 1.0, double bias = 0.0, double relu_slope = 1.0,
                        double leaky_slope = 0.01);


    struct StarReLU: xt::Module {
    public:
        StarReLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



