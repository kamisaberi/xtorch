#pragma once

#include "common.h"

namespace xt::norm
{
    struct SwitchableNorm : xt::Module
    {
    public:
        SwitchableNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
