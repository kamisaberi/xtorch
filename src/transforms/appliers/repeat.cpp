#include "include/transforms/appliers/repeat.h"

namespace xt::transforms
{
    Repeat::Repeat() = default;

    Repeat::Repeat(std::unique_ptr<xt::Module> transform, int n_times): xt::Module(), transform(std::move(transform)),
                                                          n_times_(n_times)
    {
    }

    torch::Tensor Repeat::forward(torch::Tensor input) const
    {
        for (int i = 1; i < this->n_times_; i++)
        {
            // input = transform(std::move(input));
        }
        return input;
    }
}
