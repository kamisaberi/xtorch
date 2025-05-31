#pragma once

#include "common.h"

namespace xt::norm
{
    struct OnlineNorm : xt::Module
    {
    public:
        OnlineNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
