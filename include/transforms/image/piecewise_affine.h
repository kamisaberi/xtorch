#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class PiecewiseAffine final : public xt::Module
    {
    public:
        PiecewiseAffine();
        explicit PiecewiseAffine(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
