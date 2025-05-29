#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class TemporalJitter final : public xt::Module
    {
    public:
        TemporalJitter();
        explicit TemporalJitter(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
