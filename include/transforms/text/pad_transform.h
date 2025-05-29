#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class PadTransform final : public xt::Module
    {
    public:
        PadTransform();
        explicit PadTransform(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
