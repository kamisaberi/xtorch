#include "include/transforms/weather/dynamic_shadows.h"

namespace xt::transforms::weather
{
    DynamicShadows::DynamicShadows() = default;

    DynamicShadows::DynamicShadows(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto DynamicShadows::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
