#include "include/transforms/signal/wavelet_transforms.h"

namespace xt::transforms::signal
{
    WaveletTransforms::WaveletTransforms() = default;

    WaveletTransforms::WaveletTransforms(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto WaveletTransforms::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
