#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class GridDistortion final : public xt::Module
    {
    public:
        GridDistortion();
        explicit GridDistortion(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
