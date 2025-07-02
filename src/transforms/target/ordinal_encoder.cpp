#include "include/transforms/target/ordinal_encoder.h"

namespace xt::transforms::target
{
    OrdinalEncoder::OrdinalEncoder() = default;

    OrdinalEncoder::OrdinalEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto OrdinalEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
