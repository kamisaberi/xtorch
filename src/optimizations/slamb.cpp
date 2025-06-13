#include "include/optimizations/slamb.h"
#include <stdexcept>

// --- SLAMBOptions Methods ---
void SLAMBOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta1", beta1()); archive.write("beta2", beta2());
    archive.write("eps", eps()); archive.write("weight_decay", weight_decay());
    archive.write("compression_ratio", compression_ratio());
}
void SLAMBOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    if (archive.try_read("compression_ratio", ivalue)) { compression_ratio_ = ivalue.toDouble(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> SLAMBOptions::clone() const {
    auto cloned = std::make_unique<SLAMBOptions>(this->lr());
    cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay())
          .compression_ratio(compression_ratio());
    return cloned;
}

// --- SLAMBParamState Methods ---
void SLAMBParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    if(error_feedback().defined()) archive.write("error_feedback", error_feedback(), true);
}
void SLAMBParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    if(archive.try_read("error_feedback", temp, true)) { error_feedback_ = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> SLAMBParamState::clone() const {
    auto cloned = std::make_unique<SLAMBParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    if(error_feedback().defined()) cloned->error_feedback(error_feedback().clone());
    return cloned;
}

// --- SLAMB Implementation ---
SLAMB::SLAMB(std::vector<torch::Tensor> params, SLAMBOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<SLAMBOptions>(options)) {}

SLAMB::SLAMB(std::vector<torch::Tensor> params, double lr)
    : SLAMB(std::move(params), SLAMBOptions(lr)) {}

torch::Tensor SLAMB::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<SLAMBOptions&>(param_groups_[0].options());

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            auto& state = static_cast<SLAMBParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
                state.error_feedback(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            // --- Step 1: Gradient Sparsification with Error Feedback ---
            auto compensated_grad = grad + state.error_feedback();

            int64_t k = std::max(1L, static_cast<int64_t>(p.numel() * group_options.compression_ratio()));
            auto topk_abs_values = torch::abs(compensated_grad).flatten();
            auto threshold = std::get<0>(torch::kthvalue(topk_abs_values, topk_abs_values.numel() - k)).item<float>();

            auto mask = (torch::abs(compensated_grad) >= threshold).to(grad.dtype());
            auto sparse_grad = compensated_grad * mask;

            state.error_feedback(compensated_grad - sparse_grad);

            // --- Step 2: LAMB update using the sparse_grad ---
            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            double beta1 = group_options.beta1();
            double beta2 = group_options.beta2();

            // Update biased moments using the sparse gradient
            m.mul_(beta1).add_(sparse_grad, 1.0 - beta1);
            // Note: v_t should ideally use the dense gradient for a better variance estimate.
            // Using the sparse gradient here is a simplification that makes the optimizer
            // focus its adaptation on the most important coordinates. Let's use sparse for consistency.
            v.mul_(beta2).addcmul_(sparse_grad, sparse_grad, 1.0 - beta2);

            // Bias correction
            double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
            double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);
            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;

            auto adam_update = m_hat / (v_hat.sqrt() + group_options.eps());

            // Add decoupled weight decay
            if (group_options.weight_decay() > 0.0) {
                adam_update.add_(p.detach(), group_options.weight_decay());
            }

            // Calculate Trust Ratio
            auto weight_norm = p.detach().norm(2).item<double>();
            auto update_norm = adam_update.norm(2).item<double>();

            double trust_ratio = 1.0;
            if (weight_norm > 0 && update_norm > 0) {
                trust_ratio = weight_norm / update_norm;
            }

            // Apply final update
            p.data().add_(adam_update, -group_options.lr() * trust_ratio);
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void SLAMB::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void SLAMB::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> SLAMB::make_param_state() { return std::make_unique<SLAMBParamState>(); }