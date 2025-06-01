#include "include/transforms/text/pad_transform.h"

namespace xt::transforms::text
{
    PadTransform::PadTransform() = default;

    PadTransform::PadTransform(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto PadTransform::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
