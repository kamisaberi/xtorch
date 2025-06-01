#include "include/transforms/weather/depth_based_fog.h"

namespace xt::transforms::weather
{
    DepthBasedFog::DepthBasedFog() = default;

    DepthBasedFog::DepthBasedFog(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto DepthBasedFog::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
