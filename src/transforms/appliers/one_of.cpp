
#include "include/transforms/appliers/one_of.h"

namespace xt::transforms
{

    OneOf::OneOf() = default;

    OneOf::OneOf(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }



    auto Compose::forward(std::initializer_list<torch::Tensor> tensors) -> std::any {




    }

        torch::Tensor OneOf::forward(torch::Tensor input) const
    {
        int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        input = transforms[index](std::move(input));
        return input;

    }
}
