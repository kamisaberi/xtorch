#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomApply final : public xt::Module
    {
    public:
        RandomApply();
        explicit RandomApply(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
