#pragma once

#include "../common.h"


namespace xt::transforms::weather
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
