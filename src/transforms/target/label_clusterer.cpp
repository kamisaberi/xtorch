#include "include/transforms/target/label_clusterer.h"

namespace xt::transforms::target
{
    LabelClusterer::LabelClusterer() = default;

    LabelClusterer::LabelClusterer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LabelClusterer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
