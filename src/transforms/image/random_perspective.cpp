#include "include/transforms/image/random_perspective.h"

namespace xt::transforms::image
{
    RandomPerspective::RandomPerspective() = default;

    RandomPerspective::RandomPerspective(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto RandomPerspective::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
