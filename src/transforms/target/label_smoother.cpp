#include "include/transforms/target/label_smoother.h"

namespace xt::transforms::target
{
    LabelSmoother::LabelSmoother() = default;

    LabelSmoother::LabelSmoother(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LabelSmoother::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
