#pragma once

#include "common.h"

namespace xt::norm
{
    struct SparseSwitchableNorm : xt::Module
    {
    public:
        SparseSwitchableNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
