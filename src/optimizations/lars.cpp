#include "include/optimizations/lars.h"
#include <stdexcept>

// ... (All other methods like serialize, deserialize, clone, constructors remain the same) ...

// --- LARS Implementation ---

// CORRECTED step method
torch::Tensor LARS::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    // There is only one param_group in this implementation, so we get its options.
    auto& group_options = static_cast<LARSOptions&>(param_groups_[0].options());

    // Iterate through each parameter group (though we typically have one)
    for (auto& group : param_groups_) {
        // --- THIS IS THE CORRECTED LINE ---
        // Iterate through the parameters *within the current group*.
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("LARS optimizer does not support sparse gradients.");
            }

            // --- LARS Logic ---
            // 1. Calculate norms
            auto weight_norm = p.detach().norm(2).item<double>();
            auto grad_norm = grad.norm(2).item<double>();

            // 2. Calculate local learning rate (trust ratio)
            double local_lr = 1.0;
            if (weight_norm > 0 && grad_norm > 0) {
                // The denominator is ||grad|| + wd * ||w||
                local_lr = group_options.trust_coefficient() * weight_norm /
                           (grad_norm + group_options.weight_decay() * weight_norm + group_options.eps());
            }

            // 3. Calculate SGD with Momentum update direction
            // The gradient for the momentum update includes weight decay
            auto grad_with_wd = grad.add(p.detach(), group_options.weight_decay());

            auto& state = static_cast<LARSParamState&>(*state_.at(p.unsafeGetTensorImpl()));
            if (!state.momentum_buffer().defined()) {
                state.momentum_buffer(torch::zeros_like(p));
            }
            auto& momentum_buffer = state.momentum_buffer();
            momentum_buffer.mul_(group_options.momentum()).add_(grad_with_wd);

            // 4. Apply final update
            // The effective learning rate is global_lr * local_lr
            // The update direction is from the momentum buffer
            p.data().add_(momentum_buffer, -group_options.lr() * local_lr);
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void LARS::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void LARS::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> LARS::make_param_state() { return std::make_unique<LARSParamState>(); }