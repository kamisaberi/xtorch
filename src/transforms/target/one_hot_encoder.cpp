#include "include/transforms/target/one_hot_encoder.h"

namespace xt::transforms::target
{
    OneHotEncoder::OneHotEncoder() = default;

    OneHotEncoder::OneHotEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto OneHotEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
