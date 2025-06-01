#include "include/transforms/weather/vegetation_motion.h"

namespace xt::transforms::weather
{
    VegetationMotion::VegetationMotion() = default;

    VegetationMotion::VegetationMotion(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto VegetationMotion::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
