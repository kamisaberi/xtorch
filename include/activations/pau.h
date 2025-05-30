#pragma once

#include "common.h"

namespace xt::activations
{
    torch::Tensor pau(torch::Tensor x);

    struct PAU : xt::Module
    {
    public:
        PAU() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
