
#include "include/transforms/appliers/sometimes.h"

namespace xt::transforms
{
    Sometimes::Sometimes() = default;

    Sometimes::Sometimes(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    torch::Tensor Sometimes::forward(torch::Tensor input) const
    {
        int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        input = transforms[index](std::move(input));
        return input;
    }
}
