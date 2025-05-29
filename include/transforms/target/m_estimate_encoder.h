#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class MEstimateEncoder final : public xt::Module
    {
    public:
        MEstimateEncoder();
        explicit MEstimateEncoder(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
