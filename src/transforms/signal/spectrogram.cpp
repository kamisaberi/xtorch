#include "include/transforms/signal/spectrogram.h"

namespace xt::transforms::signal
{
    Spectrogram::Spectrogram() = default;

    Spectrogram::Spectrogram(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Spectrogram::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
