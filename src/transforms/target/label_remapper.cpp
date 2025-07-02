#include "include/transforms/target/label_remapper.h"

namespace xt::transforms::target
{
    LabelRemapper::LabelRemapper() = default;

    LabelRemapper::LabelRemapper(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LabelRemapper::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
