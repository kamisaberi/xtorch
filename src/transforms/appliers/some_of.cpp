
#include "include/transforms/appliers/some_of.h"

namespace xt::transforms
{
    SomeOf::SomeOf() = default;

    SomeOf::SomeOf(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    torch::Tensor SomeOf::forward(torch::Tensor input) const
    {
        int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        input = transforms[index](std::move(input));
        return input;
    }
}
