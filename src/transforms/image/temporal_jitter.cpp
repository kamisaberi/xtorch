#include "include/transforms/image/temporal_jitter.h"

namespace xt::transforms::image
{
    TemporalJitter::TemporalJitter() = default;

    TemporalJitter::TemporalJitter(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto TemporalJitter::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
