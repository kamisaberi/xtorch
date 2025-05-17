#include <utility>

#include "transforms/appliers/compose.h"

namespace xt::transforms
{
    Compose::Compose() = default;

    Compose::Compose(std::vector<TransformFunc> transforms): xt::Module(), transforms(std::move(transforms))
    {
    }

    torch::Tensor Compose::forward(torch::Tensor input) const
    {
        for (const auto& transform : this->transforms)
        {
            input = transform(std::move(input));
        }
        return input;
    }
}
