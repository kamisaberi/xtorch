#pragma once

#include "common.h"

namespace xt::activations
{

    struct KAN : xt::Module
    {
    public:
        KAN() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
