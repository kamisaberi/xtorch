#include "include/optimizations/1_bit_adam.h"

namespace xt::optimizations
{
    OneBitAdam::OneBitAdam(std::vector<torch::Tensor>&& parameters, double lr, double momentum)
    : Optimizer(std::move(parameters), nullptr), lr_(lr), momentum_(momentum) {
        velocities_.resize(param_groups()[0].params().size());
        for (size_t i = 0; i < velocities_.size(); ++i) {
            velocities_[i] = torch::zeros_like(param_groups()[0].params()[i]);
        }
    }

    void OneBitAdam::step() {
        // auto& params = param_groups()[0].params();
        // for (size_t i = 0; i < params.size(); ++i) {
        //     auto& param = params[i];
        //     if (!param.grad().defined()) continue;
        //
        //     auto grad = param.grad();
        //     velocities_[i] = momentum_ * velocities_[i] + (1 - momentum_) * grad;
        //     param.add_(-lr_ * velocities_[i]);
        // }
    }
}
