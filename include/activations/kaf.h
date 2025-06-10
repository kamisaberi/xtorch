#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor kaf(const torch::Tensor& x,
                      const torch::Tensor& dictionary_coefs, // Shape [D] or [1, D] for broadcasting
                      const torch::Tensor& boundary_params // Shape [D-1] or [1, D-1] for broadcasting
    );

    struct KAF : xt::Module
    {
    public:
        KAF() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
