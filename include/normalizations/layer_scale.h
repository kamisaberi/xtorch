#pragma once

#include "common.h"

namespace xt::norm
{
    struct LayerScale : xt::Module
    {
    public:
        LayerScale() = default;

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}
