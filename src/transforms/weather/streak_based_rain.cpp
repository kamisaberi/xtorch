#include "include/transforms/weather/streak_based_rain.h"

namespace xt::transforms::weather
{
    StreakBasedRain::StreakBasedRain() = default;

    StreakBasedRain::StreakBasedRain(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto StreakBasedRain::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
