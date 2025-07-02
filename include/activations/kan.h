#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor b_spline_basis(const torch::Tensor& x, const torch::Tensor& grid, int k_order);
    torch::Tensor kan_spline_activation(
        const torch::Tensor& x,
        const torch::Tensor& spline_weights, // Shape (G + k - 1)
        const torch::Tensor& grid_internal, // Shape (G + 1) for G intervals
        int k_order, // e.g., 4 for cubic
        double base_activation_weight // w
    );


    struct KAN : xt::Module
    {
    public:
        KAN() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
