#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor serlu(const torch::Tensor& x, double lambda_serlu = 1.0507, double alpha_serlu = 1.67326);

    struct SERLU : xt::Module
    {
    public:
        SERLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
