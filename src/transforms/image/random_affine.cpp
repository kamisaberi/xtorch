#include "include/transforms/image/random_affine.h"

namespace xt::transforms::image
{
    RandomAffine::RandomAffine() = default;

    RandomAffine::RandomAffine(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomAffine::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
