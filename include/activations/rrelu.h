#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor rrelu(
        const torch::Tensor& x,
        double lower = 1.0 / 8.0,
        double upper = 1.0 / 3.0,
        bool training = false,
        c10::optional<at::Generator> generator = c10::nullopt
    );


    struct RReLU : xt::Module
    {
    public:
        RReLU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
