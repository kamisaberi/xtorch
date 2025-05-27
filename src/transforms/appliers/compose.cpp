#include "include/transforms/appliers/compose.h"

namespace xt::transforms
{
    Compose::Compose() = default;

    Compose::Compose(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    auto Compose::forward(std::initializer_list<torch::Tensor> tensors) -> std::any
    {
        std::vector<torch::Tensor> tensor_vec(tensors);
        torch::Tensor input = tensor_vec[0];
        for (const auto& transform : this->transforms)
        {
            input = transform(input);
        }
        return input;
    }
}
