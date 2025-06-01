#include "include/transforms/image/fancy_pca.h"

namespace xt::transforms::image
{
    FancyPCA::FancyPCA() = default;

    FancyPCA::FancyPCA(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto FancyPCA::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
