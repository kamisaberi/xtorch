#include "include/transforms/weather/falling_snow.h"

namespace xt::transforms::weather
{
    FallingSnow::FallingSnow() = default;

    FallingSnow::FallingSnow(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto FallingSnow::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
