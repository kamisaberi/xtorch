#include "include/optimizations/adafactor.h"
#include <stdexcept>

// --- AdafactorOptions Methods ---
void AdafactorOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta2", beta2()); archive.write("eps1", eps1()); archive.write("eps2", eps2());
    archive.write("clip_threshold", clip_threshold()); archive.write("weight_decay", weight_decay());
    archive.write("scale_parameter", scale_parameter()); archive.write("relative_step", relative_step());
}
void AdafactorOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("eps1", ivalue)) { eps1_ = ivalue.toDouble(); }
    if (archive.try_read("eps2", ivalue)) { eps2_ = ivalue.toDouble(); }
    if (archive.try_read("clip_threshold", ivalue)) { clip_threshold_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    if (archive.try_read("scale_parameter", ivalue)) { scale_parameter_ = ivalue.toBool(); }
    if (archive.try_read("relative_step", ivalue)) { relative_step_ = ivalue.toBool(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> AdafactorOptions::clone() const {
    // Create a new options object. The constructor will handle the optional lr.
    // We can just pass the current lr() value.
    auto cloned = std::make_unique<AdafactorOptions>();

    // The lr() getter always returns a double. We can directly set it.
    cloned->lr(this->lr());

    // Copy the rest of the parameters.
    cloned->beta2(beta2())
          .eps1(eps1())
          .eps2(eps2())
          .clip_threshold(clip_threshold())
          .weight_decay(weight_decay())
          .scale_parameter(scale_parameter())
          .relative_step(relative_step());

    return cloned;
}

// --- AdafactorParamState Methods ---
void AdafactorParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg_sq_row.defined()) archive.write("exp_avg_sq_row", exp_avg_sq_row, true);
    if(exp_avg_sq_col.defined()) archive.write("exp_avg_sq_col", exp_avg_sq_col, true);
}
void AdafactorParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg_sq_row", temp, true)) { exp_avg_sq_row = temp; }
    if(archive.try_read("exp_avg_sq_col", temp, true)) { exp_avg_sq_col = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> AdafactorParamState::clone() const {
    auto cloned = std::make_unique<AdafactorParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg_sq_row.defined()) cloned->exp_avg_sq_row = exp_avg_sq_row.clone();
    if(exp_avg_sq_col.defined()) cloned->exp_avg_sq_col = exp_avg_sq_col.clone();
    return cloned;
}


// --- Adafactor Implementation ---
Adafactor::Adafactor(std::vector<torch::Tensor> params, AdafactorOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<AdafactorOptions>(options)) {}

double Adafactor::_get_relative_step_size(const torch::Tensor& param, long step) {
    auto& group_options = static_cast<AdafactorOptions&>(param_groups_[0].options());
    double relative_step_size = group_options.lr(); // Use fixed LR if specified

    if (group_options.relative_step()) {
        double warmup_init = std::pow(step, -0.5);
        relative_step_size = std::min(warmup_init, 1.0 / std::sqrt(step));
    }

    if (group_options.scale_parameter()) {
        relative_step_size *= std::max(1e-3, (double)param.norm().item<double>());
    }

    return relative_step_size;
}


torch::Tensor Adafactor::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<AdafactorOptions&>(param_groups_[0].options());

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("Adafactor optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<AdafactorParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                // Initialize based on parameter shape
                if (p.dim() >= 2) {
                    state.exp_avg_sq_row = torch::zeros({p.size(0)}, p.options());
                    state.exp_avg_sq_col = torch::zeros({p.size(1)}, p.options());
                } else {
                    state.exp_avg_sq_row = torch::zeros_like(p); // Fallback for 1D
                }
            }
            state.step(state.step() + 1.0);
            long current_step_val = static_cast<long>(state.step().item<double>());

            // Get learning rate for this step
            double step_size = _get_relative_step_size(p, current_step_val);

            // Regularize the squared gradient
            auto grad_sq = grad.square().add(group_options.eps1());

            // Bias correction term for beta2
            double bias_correction2 = 1.0 - std::pow(group_options.beta2(), current_step_val);

            torch::Tensor update;

            if (p.dim() >= 2) {
                // Factor-wise update for 2D+ tensors
                auto& R = state.exp_avg_sq_row;
                auto& C = state.exp_avg_sq_col;

                // Update factor EMAs
                R.mul_(group_options.beta2()).add_(grad_sq.mean(1), 1.0 - group_options.beta2());
                C.mul_(group_options.beta2()).add_(grad_sq.mean(0), 1.0 - group_options.beta2());

                // Approximate v_hat with factors
                auto R_hat = R / bias_correction2;
                auto C_hat = C / bias_correction2;

                auto R_inv_sqrt = R_hat.rsqrt().unsqueeze(1); // [M, 1]
                auto C_inv_sqrt = C_hat.rsqrt().unsqueeze(0); // [1, N]

                // This is an outer product approximation of the inverse sqrt of v_hat
                auto denom = R_inv_sqrt * C_inv_sqrt;

                update = grad * denom;

            } else {
                // Fallback to Adam-like update for 1D tensors
                auto& v = state.exp_avg_sq_row; // Reuse row accumulator for 1D
                v.mul_(group_options.beta2()).add_(grad_sq, 1.0 - group_options.beta2());
                auto v_hat = v / bias_correction2;
                update = grad / (v_hat.sqrt() + group_options.eps2());
            }

            // Clip update and apply weight decay
            auto update_norm = update.norm() / std::sqrt(update.numel());
            if ((update_norm > group_options.clip_threshold()).item<bool>()) {
                update.mul_(group_options.clip_threshold() / update_norm);
            }

            if (group_options.weight_decay() > 0.0) {
                p.data().add_(p.data(), -step_size * group_options.weight_decay());
            }

            p.data().add_(update, -step_size);
        }
    }
    return loss;
}


// --- Boilerplate Methods ---
void Adafactor::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void Adafactor::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> Adafactor::make_param_state() { return std::make_unique<AdafactorParamState>(); }