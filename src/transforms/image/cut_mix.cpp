#include "include/transforms/image/cut_mix.h"

namespace xt::transforms::image
{
    CutMix::CutMix() = default;

    CutMix::CutMix(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto CutMix::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
