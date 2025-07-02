#include "include/transforms/image/ada.h"

namespace xt::transforms::image
{
    ADA::ADA() = default;

    ADA::ADA(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto ADA::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
