#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::video
{
    class RandomClipReverse final : public xt::Module
    {
    public:
        RandomClipReverse();
        explicit RandomClipReverse(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
