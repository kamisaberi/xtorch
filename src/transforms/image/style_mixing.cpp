#include "include/transforms/image/style_mixing.h"

namespace xt::transforms::image
{
    StyleMixing::StyleMixing() = default;

    StyleMixing::StyleMixing(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto StyleMixing::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
