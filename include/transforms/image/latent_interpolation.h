#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    class LatentInterpolation final : public xt::Module
    {
    public:
        LatentInterpolation();
        explicit LatentInterpolation(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
