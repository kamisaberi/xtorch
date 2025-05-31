#pragma once

#include "common.h"

namespace xt::norm
{
    struct ModeNorm : xt::Module
    {
    public:
        ModeNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
