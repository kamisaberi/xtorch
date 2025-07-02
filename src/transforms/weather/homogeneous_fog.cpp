#include "include/transforms/weather/homogeneous_fog.h"

namespace xt::transforms::weather
{
    HomogeneousFog::HomogeneousFog() = default;

    HomogeneousFog::HomogeneousFog(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto HomogeneousFog::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
