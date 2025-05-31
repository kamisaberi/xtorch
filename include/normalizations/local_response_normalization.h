#pragma once

#include "common.h"

namespace xt::norm
{
    struct LocalResponseNorm : xt::Module
    {
    public:
        LocalResponseNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
