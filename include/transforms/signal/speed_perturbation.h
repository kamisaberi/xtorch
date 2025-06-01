#pragma once

#include "include/transforms/common.h"

namespace xt::transforms::signal
{
    class SpeedPerturbation final : public xt::Module
    {
    public:
        SpeedPerturbation();
        explicit SpeedPerturbation(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
