#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class RandomScale final : public xt::Module
    {
    public:
        RandomScale();
        explicit RandomScale(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
