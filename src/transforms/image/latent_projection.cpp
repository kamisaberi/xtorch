#include "include/transforms/image/latent_projection.h"

namespace xt::transforms::image
{
    LatentProjection::LatentProjection() = default;

    LatentProjection::LatentProjection(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LatentProjection::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
