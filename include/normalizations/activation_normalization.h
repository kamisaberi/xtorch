#pragma once

#include "common.h"

namespace xt::norm
{
    struct EvoNorm : xt::Module
    {
    public:
        EvoNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
