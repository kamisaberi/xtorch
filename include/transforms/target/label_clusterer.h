#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::target
{
    class LabelClusterer final : public xt::Module
    {
    public:
        LabelClusterer();
        explicit LabelClusterer(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
