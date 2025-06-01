#include "include/transforms/target/frequency_encoder.h"

namespace xt::transforms::target
{
    FrequencyEncoder::FrequencyEncoder() = default;

    FrequencyEncoder::FrequencyEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto FrequencyEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
