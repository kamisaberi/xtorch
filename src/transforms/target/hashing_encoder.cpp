#include "include/transforms/target/hashing_encoder.h"

namespace xt::transforms::target
{
    HashingEncoder::HashingEncoder() = default;

    HashingEncoder::HashingEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto HashingEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
