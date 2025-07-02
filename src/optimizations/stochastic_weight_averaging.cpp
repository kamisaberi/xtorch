#include "include/optimizations/stochastic_weight_averaging.h"
#include <stdexcept>

namespace xt::optim
{
    // --- StochasticWeightAveraging Implementation ---

    StochasticWeightAveraging::StochasticWeightAveraging(
        std::unique_ptr<torch::optim::Optimizer> base_optimizer,
        int swa_start_step,
        int swa_update_frequency)
        : base_optimizer_(std::move(base_optimizer)),
          swa_start_step_(swa_start_step),
          swa_update_frequency_(swa_update_frequency) {

        TORCH_CHECK(base_optimizer_ != nullptr, "A valid base optimizer must be provided.");
        TORCH_CHECK(swa_start_step_ >= 0, "SWA start step must be non-negative.");

        // Initialize the SWA parameter buffers by cloning the initial model weights
        for (const auto& group : base_optimizer_->param_groups()) {
            for (const auto& p : group.params()) {
                swa_params_.push_back(p.detach().clone());
            }
        }
    }

    void StochasticWeightAveraging::zero_grad() {
        base_optimizer_->zero_grad();
    }

    torch::Tensor StochasticWeightAveraging::step(torch::optim::Optimizer::LossClosure closure) {
        // 1. Perform a step with the base optimizer (e.g., SGD)
        torch::Tensor loss = base_optimizer_->step(closure);
        global_step_++;

        // 2. Check if we are in the SWA phase and if it's time to update the average
        if (global_step_ > swa_start_step_ && (global_step_ - swa_start_step_) % swa_update_frequency_ == 0) {

            // Update the running average of the weights
            n_averaged_++;

            int param_idx = 0;
            for (const auto& group : base_optimizer_->param_groups()) {
                for (const auto& p : group.params()) {
                    // SWA update rule: avg_n = (n-1)/n * avg_{n-1} + 1/n * w_n
                    // which is equivalent to: avg_n = avg_{n-1} + (w_n - avg_{n-1}) / n
                    auto& swa_p = swa_params_[param_idx];
                    swa_p.add_(p.detach() - swa_p, 1.0 / static_cast<double>(n_averaged_));
                    param_idx++;
                }
            }
        }

        return loss;
    }

    void StochasticWeightAveraging::swap_swa_weights() {
        // If no averaging has been done, there's nothing to swap.
        if (n_averaged_ == 0) {
            TORCH_WARN("SWA::swap_swa_weights() called before any averaging was done. No weights were swapped.");
            return;
        }

        torch::NoGradGuard no_grad;
        int param_idx = 0;
        for (auto& group : base_optimizer_->param_groups()) {
            for (auto& p : group.params()) {
                // Copy the averaged weight into the model's parameter tensor
                p.data().copy_(swa_params_[param_idx]);
                param_idx++;
            }
        }
    }
}