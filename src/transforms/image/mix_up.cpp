#include "include/transforms/image/mix_up.h"

namespace xt::transforms::image
{
    MixUp::MixUp() = default;

    MixUp::MixUp(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MixUp::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
