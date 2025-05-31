#include "include/optimizations/ao.h"

namespace xt::optimizations
{
    AO::AO(std::vector<torch::Tensor>&& parameters, double lr, double momentum)
        : Optimizer(std::move(parameters), nullptr), lr_(lr), momentum_(momentum)
    {
        velocities_.resize(param_groups()[0].params().size());
        for (size_t i = 0; i < velocities_.size(); ++i)
        {
            velocities_[i] = torch::zeros_like(param_groups()[0].params()[i]);
        }
    }

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor AO::step(LossClosure closure)
    {
        return torch::Tensor();
    }
}
