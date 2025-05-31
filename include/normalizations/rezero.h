#pragma once

#include "common.h"

namespace xt::norm
{
    struct Rezero : xt::Module
    {
    public:
        Rezero() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
