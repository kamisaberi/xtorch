
#include "include/transforms/appliers/random_apply.h"

namespace xt::transforms
{

    RandomApply::RandomApply() = default;

    RandomApply::RandomApply(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }


    torch::Tensor RandomApply::forward(torch::Tensor input) const
    {
        int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        input = transforms[index](std::move(input));
        return input;
    }


    auto Compose::forward(std::initializer_list<torch::Tensor> tensors) -> std::any {

        int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        input = transforms[index](std::move(input));
        return input;


    }



    }
