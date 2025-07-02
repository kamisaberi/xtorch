#include "include/transforms/target/target_normalizer.h"

namespace xt::transforms::target
{
    TargetNormalizer::TargetNormalizer() = default;

    TargetNormalizer::TargetNormalizer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto TargetNormalizer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
