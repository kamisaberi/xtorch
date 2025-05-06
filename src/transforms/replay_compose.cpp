#include <utility>

#include "../../include/transforms/replay_compose.h"

namespace xt::transforms
{
    using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;

    ReplayCompose::ReplayCompose() = default;

    ReplayCompose::ReplayCompose(std::vector<TransformFunc> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    torch::Tensor ReplayCompose::operator()(torch::Tensor input) const
    {
        int index = torch::randint(0, transforms.size() - 1, {}, torch::kInt32).item<int>();
        input = transforms[index](std::move(input));
        return input;
    }
}
