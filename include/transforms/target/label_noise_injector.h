#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class LabelNoiseInjector final : public xt::Module
    {
    public:
        LabelNoiseInjector();
        explicit LabelNoiseInjector(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
