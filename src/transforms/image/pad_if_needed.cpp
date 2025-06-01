#include "include/transforms/image/pad_if_needed.h"

namespace xt::transforms::image
{
    PadIfNeeded::PadIfNeeded() = default;

    PadIfNeeded::PadIfNeeded(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto PadIfNeeded::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
