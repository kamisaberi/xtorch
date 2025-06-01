#include "include/transforms/target/power_transformer.h"

namespace xt::transforms::target
{
    PowerTransformer::PowerTransformer() = default;

    PowerTransformer::PowerTransformer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto PowerTransformer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
