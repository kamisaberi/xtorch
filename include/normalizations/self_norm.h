#pragma once

#include "common.h"

namespace xt::norm
{
    struct SelfNorm : xt::Module
    {
    public:
        SelfNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
