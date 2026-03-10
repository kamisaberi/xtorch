#pragma once

#include "common.h"

namespace xt::activations
{

    struct KAF : xt::Module
    {
    public:
        KAF() = default;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
