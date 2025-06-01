#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class LatentProjection final : public xt::Module
    {
    public:
        LatentProjection();
        explicit LatentProjection(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
