#pragma once

#include "common.h"

namespace xt::norm
{
    struct MixtureNorm : xt::Module
    {
    public:
        MixtureNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
