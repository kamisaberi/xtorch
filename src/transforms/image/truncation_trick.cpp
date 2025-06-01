#include "include/transforms/image/truncation_trick.h"

namespace xt::transforms::image
{
    TruncationTrick::TruncationTrick() = default;

    TruncationTrick::TruncationTrick(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto TruncationTrick::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
