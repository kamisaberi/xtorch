#include "include/transforms/target/target_encoder.h"

namespace xt::transforms::target
{
    TargetEncoder::TargetEncoder() = default;

    TargetEncoder::TargetEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto TargetEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
