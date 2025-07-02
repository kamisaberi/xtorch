#include "include/transforms/target/binary_encoder.h"

namespace xt::transforms::target
{
    BinaryEncoder::BinaryEncoder() = default;

    BinaryEncoder::BinaryEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto BinaryEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
