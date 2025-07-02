#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor evonorm_s0(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta,
                             const torch::Tensor& v_param, int64_t num_groups,
                             double eps = 1e-5);

    struct EvonormS0 : xt::Module
    {
    public:
        EvonormS0() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
