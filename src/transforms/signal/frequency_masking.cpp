#include "include/transforms/signal/frequency_masking.h"

namespace xt::transforms::signal
{
    FrequencyMasking::FrequencyMasking() = default;

    FrequencyMasking::FrequencyMasking(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto FrequencyMasking::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
