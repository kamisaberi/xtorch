#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class NoiseInjection final : public xt::Module
    {
    public:
        NoiseInjection();
        explicit NoiseInjection(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
