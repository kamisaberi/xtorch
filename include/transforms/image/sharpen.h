#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class Sharpen final : public xt::Module
    {
    public:
        Sharpen();
        explicit Sharpen(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
