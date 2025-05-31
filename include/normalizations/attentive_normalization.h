#pragma once

#include "common.h"

namespace xt::norm
{
    struct AttentiveNorm : xt::Module
    {
    public:
        AttentiveNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
