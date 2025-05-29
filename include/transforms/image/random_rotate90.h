#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomRotate90 final : public xt::Module
    {
    public:
        RandomRotate90();
        explicit RandomRotate90(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
