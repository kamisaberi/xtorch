#include <optimizations/yellow_fin.h>
#include <stdexcept>

namespace xt::optim
{
    // --- YellowFinOptions Methods ---
    void YellowFinOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta", beta());
        archive.write("weight_decay", weight_decay());
        archive.write("curv_min_clamp", curv_min_clamp());
        archive.write("curv_max_clamp", curv_max_clamp());
    }
    void YellowFinOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta", ivalue)) { beta_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("curv_min_clamp", ivalue)) { curv_min_clamp_ = ivalue.toDouble(); }
        if (archive.try_read("curv_max_clamp", ivalue)) { curv_max_clamp_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> YellowFinOptions::clone() const {
        auto cloned = std::make_unique<YellowFinOptions>(this->lr());
        cloned->beta(beta()).weight_decay(weight_decay())
              .curv_min_clamp(curv_min_clamp()).curv_max_clamp(curv_max_clamp());
        return cloned;
    }

    // --- YellowFinParamState Methods ---
    void YellowFinParamState::serialize(torch::serialize::OutputArchive& archive) const {
        if(momentum_buffer().defined()) archive.write("momentum_buffer", momentum_buffer(), true);
    }
    void YellowFinParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("momentum_buffer", temp, true)) { momentum_buffer_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> YellowFinParamState::clone() const {
        auto cloned = std::make_unique<YellowFinParamState>();
        if(momentum_buffer().defined()) cloned->momentum_buffer(momentum_buffer().clone());
        return cloned;
    }


    // --- YellowFin Implementation ---
    YellowFin::YellowFin(std::vector<torch::Tensor> params, YellowFinOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<YellowFinOptions>(options)) {}

    torch::Tensor YellowFin::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<YellowFinOptions&>(param_groups_[0].options());
        step_count_++;

        // --- 1. Flatten all gradients and parameters for global measurements ---
        std::vector<torch::Tensor> all_grads, all_params;
        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (p.grad().defined()) {
                    all_grads.push_back(p.grad().detach().flatten());
                    all_params.push_back(p.detach().flatten());
                }
            }
        }
        if (all_grads.empty()) { return loss; }
        auto grad_flat = torch::cat(all_grads);
        auto param_flat = torch::cat(all_params);

        // --- 2. Measure instantaneous curvature and gradient variance ---
        double h_min_inst = 0.0, h_max_inst = 0.0;
        if (step_count_ > 1) {
            auto grad_dist_sq = torch::sum((grad_flat - global_state_.prev_grad_flat).square()).item<double>();
            auto param_dist_sq = torch::sum((param_flat - global_state_.prev_param_flat).square()).item<double>();

            // Approximate h_max using the single direction of change
            double curvature = (param_dist_sq > 1e-12) ? grad_dist_sq / param_dist_sq : 0.0;

            // In a single direction, h_min = h_max. We need a more robust estimate.
            // For simplicity, we'll use the gradient norm as a proxy for min curvature,
            // as done in some practical implementations.
            h_max_inst = curvature;
            h_min_inst = grad_flat.square().mean().item<double>(); // Proxy for smallest curvature
        }

        // Clamp curvature estimates for stability
        h_max_inst = std::max(h_max_inst, group_options.curv_min_clamp());
        h_min_inst = std::max(h_min_inst, group_options.curv_min_clamp());
        h_max_inst = std::min(h_max_inst, group_options.curv_max_clamp());
        h_min_inst = std::min(h_min_inst, h_max_inst);

        double grad_var_inst = grad_flat.square().mean().item<double>();

        // Update state for next step
        global_state_.prev_grad_flat = grad_flat.clone();
        global_state_.prev_param_flat = param_flat.clone();

        // --- 3. Smooth measurements with EMA ---
        double beta = group_options.beta();
        global_state_.h_min_ema = beta * global_state_.h_min_ema + (1 - beta) * h_min_inst;
        global_state_.h_max_ema = beta * global_state_.h_max_ema + (1 - beta) * h_max_inst;
        global_state_.grad_var_ema = beta * global_state_.grad_var_ema + (1 - beta) * grad_var_inst;

        // --- 4. Auto-tune Learning Rate and Momentum ---
        double p = global_state_.h_max_ema / global_state_.h_min_ema;
        double C = global_state_.grad_var_ema;

        double mu_num = std::pow(std::sqrt(p) - 1, 2);
        double mu_den = std::pow(std::sqrt(p) + 1, 2);
        global_state_.tuned_momentum = mu_num / mu_den;

        double lr_num = std::pow(1 - std::sqrt(global_state_.tuned_momentum), 2);
        double lr_den = global_state_.h_min_ema;
        global_state_.tuned_lr = lr_num / lr_den;

        // --- 5. Apply the update with tuned hyperparameters ---
        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }
                auto& state = static_cast<YellowFinParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.momentum_buffer().defined()) {
                    state.momentum_buffer(torch::zeros_like(p));
                }
                auto& momentum_buffer = state.momentum_buffer();

                // Apply weight decay to the gradient
                auto grad = p.grad();
                if (group_options.weight_decay() > 0.0) {
                    grad = grad.add(p.detach(), group_options.weight_decay());
                }

                // Standard momentum update using the *tuned* momentum
                momentum_buffer.mul_(global_state_.tuned_momentum).add_(grad);

                // Final update using the *tuned* learning rate
                p.data().add_(momentum_buffer, -global_state_.tuned_lr);
            }
        }

        return loss;
    }

    // --- Boilerplate ---
    void YellowFin::save(torch::serialize::OutputArchive& archive) const { /* Custom state saving needed */ }
    void YellowFin::load(torch::serialize::InputArchive& archive) { /* Custom state loading needed */ }
    std::unique_ptr<torch::optim::OptimizerParamState> YellowFin::make_param_state() { return std::make_unique<YellowFinParamState>(); }
}