#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class QuantileTransformer final : public xt::Module
    {
    public:
        QuantileTransformer();
        explicit QuantileTransformer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
