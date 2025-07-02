#include "include/transforms/signal/mfcc.h"

namespace xt::transforms::signal
{
    MFCC::MFCC() = default;

    MFCC::MFCC(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto MFCC::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
