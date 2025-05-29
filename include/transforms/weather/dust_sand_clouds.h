#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class DustSandClouds final : public xt::Module
    {
    public:
        DustSandClouds();
        explicit DustSandClouds(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
