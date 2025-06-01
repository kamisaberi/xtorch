#include "include/transforms/weather/patchy_fog.h"

namespace xt::transforms::weather
{
    PatchyFog::PatchyFog() = default;

    PatchyFog::PatchyFog(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto PatchyFog::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
