#pragma once

#include "common.h"

namespace xt::norm
{
    struct BatchChannelNorm : xt::Module
    {
    public:
        BatchChannelNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
