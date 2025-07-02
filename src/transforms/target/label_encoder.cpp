#include "include/transforms/target/label_encoder.h"

namespace xt::transforms::target
{
    LabelEncoder::LabelEncoder() = default;

    LabelEncoder::LabelEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto LabelEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
