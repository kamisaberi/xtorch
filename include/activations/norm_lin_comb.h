#pragma once

#include "common.h"

namespace xt::activations
{


    struct NormLinComb : xt::Module
    {
    public:
        NormLinComb() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
