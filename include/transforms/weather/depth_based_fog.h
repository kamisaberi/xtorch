#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class DepthBasedFog final : public xt::Module
    {
    public:
        DepthBasedFog();
        explicit DepthBasedFog(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
