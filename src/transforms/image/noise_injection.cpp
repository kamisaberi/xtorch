#include "include/transforms/image/noise_injection.h"

namespace xt::transforms::image
{
    NoiseInjection::NoiseInjection() = default;

    NoiseInjection::NoiseInjection(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto NoiseInjection::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
