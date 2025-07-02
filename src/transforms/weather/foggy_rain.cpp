#include "include/transforms/weather/foggy_rain.h"

namespace xt::transforms::weather
{
    FoggyRain::FoggyRain() = default;

    FoggyRain::FoggyRain(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto FoggyRain::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
