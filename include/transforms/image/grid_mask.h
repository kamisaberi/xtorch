#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class GridMask final : public xt::Module
    {
    public:
        GridMask();
        explicit GridMask(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
