#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class RandomMosaic final : public xt::Module
    {
    public:
        RandomMosaic();
        explicit RandomMosaic(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
