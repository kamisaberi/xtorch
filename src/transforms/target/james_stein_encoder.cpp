#include "include/transforms/target/james_stein_encoder.h"

namespace xt::transforms::target
{
    JamesSteinEncoder::JamesSteinEncoder() = default;

    JamesSteinEncoder::JamesSteinEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto JamesSteinEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
