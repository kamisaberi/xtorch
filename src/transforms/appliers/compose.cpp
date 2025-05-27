#include "include/transforms/appliers/compose.h"

namespace xt::transforms
{
    Compose::Compose() = default;

    Compose::Compose(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    auto Compose::forward(std::initializer_list<torch::Tensor> tensors) -> std::any
    {
        auto input = tensors;
        for (const auto& transform : this->transforms)
        {
            input = transform(input);
        }
        return input;
    }
}
