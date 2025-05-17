#include <utility>

#include "transforms/appliers/replay_compose.h"

namespace xt::transforms
{

    ReplayCompose::ReplayCompose() = default;

    ReplayCompose::ReplayCompose(std::vector<xt::Module> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    torch::Tensor ReplayCompose::forward(torch::Tensor input) const
    {
        int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        input = transforms[index](std::move(input));
        return input;
    }
}
