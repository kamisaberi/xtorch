#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class MaskDropout final : public xt::Module
    {
    public:
        MaskDropout();
        explicit MaskDropout(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
