#include "include/transforms/image/latent_interpolation.h"

namespace xt::transforms::image
{
    LatentInterpolation::LatentInterpolation() = default;

    LatentInterpolation::LatentInterpolation(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LatentInterpolation::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
