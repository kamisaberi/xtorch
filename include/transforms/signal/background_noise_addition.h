#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::signal
{
    class BackgroundNoiseAddition final : public xt::Module
    {
    public:
        BackgroundNoiseAddition();
        explicit BackgroundNoiseAddition(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
