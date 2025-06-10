#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor srelu(
    const torch::Tensor& x,
    const torch::Tensor& t_left, // Threshold for left part
    const torch::Tensor& a_left, // Slope for left part
    const torch::Tensor& t_right, // Threshold for right part
    const torch::Tensor& a_right // Slope for right part
);


    struct SReLU : xt::Module {
    public:
        SReLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



