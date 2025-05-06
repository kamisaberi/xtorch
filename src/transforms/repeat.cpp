#include "../../include/transforms/repeat.h"

namespace xt::transforms
{
    using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;

    Repeat::Repeat() = default;

    Repeat::Repeat(TransformFunc transform, int n_times): xt::Module(), transform(std::move(transform)),
                                                          n_times_(n_times)
    {
    }

    torch::Tensor Repeat::operator()(torch::Tensor input) const
    {
        for (int i = 1; i < this->n_times_; i++)
        {
            input = transform(std::move(input));
        }
        return input;
    }
}
