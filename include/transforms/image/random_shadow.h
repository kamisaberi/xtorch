#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class RandomShadow final : public xt::Module
    {
    public:
        RandomShadow();
        explicit RandomShadow(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
