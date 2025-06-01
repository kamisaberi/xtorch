#include "include/transforms/image/grid_distortion.h"

namespace xt::transforms::image
{
    GridDistortion::GridDistortion() = default;

    GridDistortion::GridDistortion(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto GridDistortion::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
