#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class MixUp final : public xt::Module
    {
    public:
        MixUp();
        explicit MixUp(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
