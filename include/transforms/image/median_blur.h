#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class MedianBlur final : public xt::Module
    {
    public:
        MedianBlur();
        explicit MedianBlur(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
