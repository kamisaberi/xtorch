#include "include/transforms/signal/resample.h"

namespace xt::transforms::signal
{
    Resample::Resample() = default;

    Resample::Resample(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Resample::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
