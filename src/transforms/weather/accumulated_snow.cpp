#include "include/transforms/weather/accumulated_snow.h"

namespace xt::transforms::weather
{
    AccumulatedSnow::AccumulatedSnow() = default;

    AccumulatedSnow::AccumulatedSnow(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto AccumulatedSnow::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
