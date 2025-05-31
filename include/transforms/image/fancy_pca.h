#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class FancyPCA final : public xt::Module
    {
    public:
        FancyPCA();
        explicit FancyPCA(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
