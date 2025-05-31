#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class RandomCropNearBbox final : public xt::Module
    {
    public:
        RandomCropNearBbox();
        explicit RandomCropNearBbox(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
