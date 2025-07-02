#include "include/transforms/text/truncate.h"

namespace xt::transforms::text
{
    Truncate::Truncate() = default;

    Truncate::Truncate(std::vector<xt::Module> transforms) : xt::Module()
    {
    }

    auto Truncate::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
