#pragma once

#include "include/transforms/common.h"

namespace xt::transforms
{
    class StreakBasedRain final : public xt::Module
    {
    public:
        StreakBasedRain();
        explicit StreakBasedRain(std::vector<xt::Module> transforms);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}
