#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::signal
{
    class Resample final : public xt::Module
    {
    public:
        Resample();
        explicit Resample(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
