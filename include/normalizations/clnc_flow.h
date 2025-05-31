#pragma once

#include "common.h"

namespace xt::norm
{
    struct CLNCFlow : xt::Module
    {
    public:
        CLNCFlow() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
