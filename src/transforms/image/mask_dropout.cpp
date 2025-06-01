#include "include/transforms/image/mask_dropout.h"

namespace xt::transforms::image
{
    MaskDropout::MaskDropout() = default;

    MaskDropout::MaskDropout(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MaskDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
