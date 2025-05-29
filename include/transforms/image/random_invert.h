#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomInvert final : public xt::Module
    {
    public:
        RandomInvert();
        explicit RandomInvert(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
