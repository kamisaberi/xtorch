#include "include/transforms/target/binarizer.h"

namespace xt::transforms::target
{
    Binarizer::Binarizer() = default;

    Binarizer::Binarizer(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Binarizer::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
