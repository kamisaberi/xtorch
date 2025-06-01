#include "include/transforms/target/m_estimate_encoder.h"

namespace xt::transforms::target
{
    MEstimateEncoder::MEstimateEncoder() = default;

    MEstimateEncoder::MEstimateEncoder(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MEstimateEncoder::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
