#pragma once

#include "common.h"

namespace xt::activations {
    torch::Tensor nfn(
        const torch::Tensor& x,
        const torch::Tensor& alpha, // Shape (num_filters_out, num_filters_in, filter_size)
        const torch::Tensor& beta // Shape (num_filters_out, num_filters_in, filter_size)
    );

    struct NFN : xt::Module {
    public:
        NFN() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}



