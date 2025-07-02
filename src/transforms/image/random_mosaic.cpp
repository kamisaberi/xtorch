#include "include/transforms/image/random_mosaic.h"

namespace xt::transforms::image
{
    RandomMosaic::RandomMosaic() = default;

    RandomMosaic::RandomMosaic(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomMosaic::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
