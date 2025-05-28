#include "include/transforms/appliers/compose.h"

namespace xt::transforms
{
    Compose::Compose() = default;

    Compose::Compose(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    auto Compose::forward(std::initializer_list<std::any> tensors) -> std::any
    {


        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensors_vec =
        torch::Tensor input = tensor_vec[0];
        for (auto& transform : this->transforms)
        {
            input = std::any_cast<torch::Tensor>(transform({input}));
        }
        return input;
    }
}
