#pragma once

#include "common.h"

namespace xt::norm
{
    struct SpectralNorm : xt::Module
    {
    public:
        SpectralNorm() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
