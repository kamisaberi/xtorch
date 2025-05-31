#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::image
{
    class PadIfNeeded final : public xt::Module
    {
    public:
        PadIfNeeded();
        explicit PadIfNeeded(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
