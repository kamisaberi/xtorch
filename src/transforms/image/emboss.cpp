#include "include/transforms/image/emboss.h"

namespace xt::transforms::image
{
    Emboss::Emboss() = default;

    Emboss::Emboss(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Emboss::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
