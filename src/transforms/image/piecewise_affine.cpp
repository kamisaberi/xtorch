#include "include/transforms/image/piecewise_affine.h"

namespace xt::transforms::image
{
    PiecewiseAffine::PiecewiseAffine() = default;

    PiecewiseAffine::PiecewiseAffine(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto PiecewiseAffine::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
