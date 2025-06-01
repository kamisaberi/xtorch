#include "include/transforms/signal/mel_spectrogram.h"

namespace xt::transforms::signal
{
    MelSpectrogram::MelSpectrogram() = default;

    MelSpectrogram::MelSpectrogram(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MelSpectrogram::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
