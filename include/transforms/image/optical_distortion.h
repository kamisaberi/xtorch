#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class OpticalDistortion final : public xt::Module
    {
    public:
        OpticalDistortion();
        explicit OpticalDistortion(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
