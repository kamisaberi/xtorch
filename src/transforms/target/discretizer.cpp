#include "include/transforms/target/discretizer.h"

namespace xt::transforms::target
{
    Discretizer::Discretizer() = default;

    Discretizer::Discretizer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Discretizer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
