#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class Upscale final : public xt::Module
    {
    public:
        Upscale();
        explicit Upscale(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
