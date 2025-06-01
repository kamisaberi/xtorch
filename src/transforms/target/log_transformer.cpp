#include "include/transforms/target/log_transformer.h"

namespace xt::transforms::target
{
    LogTransformer::LogTransformer() = default;

    LogTransformer::LogTransformer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LogTransformer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
