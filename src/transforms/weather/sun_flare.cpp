#include "include/transforms/weather/sun_flare.h"

namespace xt::transforms::weather
{
    SunFlare::SunFlare() = default;

    SunFlare::SunFlare(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto SunFlare::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
