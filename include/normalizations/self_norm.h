#pragma once

#include "common.h"

namespace xt::norm
{
    struct SelfNorm : xt::Module
    {
    public:
        SelfNorm(bool inplace = false);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // SELU has fixed alpha and scale parameters.
        // These are constants derived for self-normalizing properties.
        // alpha = 1.6732632423543772848170429916717
        // scale = 1.0507009873554804934193349852946
        // LibTorch's torch::selu function uses these constants internally.

        bool inplace_; // Whether to perform the operation in-place

    };
}
