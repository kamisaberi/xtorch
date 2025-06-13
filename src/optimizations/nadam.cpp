#include "include/optimizations/nadam.h"
#include <stdexcept>

// --- NAdamOptions Methods ---
void NAdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta1", beta1());
    archive.write("beta2", beta2());
    archive.write("eps", eps());
    archive.write("weight_decay", weight_decay());
}

void NAdamOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
}

std::unique_ptr<torch::optim::OptimizerOptions> NAdamOptions::clone() const {
    auto cloned = std::make_unique<NAdamOptions>(this->lr());
    cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
    return cloned;
}

// --- NAdamParamState Methods ---
void NAdamParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
}

void NAdamParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
}

std::unique_ptr<torch::optim::OptimizerParamState> NAdamParamState::clone() const {
    auto cloned = std::make_unique<NAdamParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    return cloned;
}


// --- NAdam Implementation ---
NAdam::NAdam(std::vector<torch::Tensor> params, NAdamOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<NAdamOptions>(options)) {}

NAdam::NAdam(std::vector<torch::Tensor> params, double lr)
    : NAdam(std::move(params), NAdamOptions(lr)) {}

torch::Tensor NAdam::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<NAdamOptions&>(param_groups_[0].options());

    for (auto& p : param_groups_[0].params()) {
        if (!p.grad().defined()) { continue; }

        auto grad = p.grad();
        if (grad.is_sparse()) {
            throw std::runtime_error("NAdam optimizer does not support sparse gradients.");
        }

        auto& state = static_cast<NAdamParamState&>(*state_.at(p.unsafeGetTensorImpl()));

        if (!state.step().defined()) {
            state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
            state.exp_avg(torch::zeros_like(p));
            state.exp_avg_sq(torch::zeros_like(p));
        }
        state.step(state.step() + 1.0);
        double current_step_val = state.step().item<double>();

        auto& m = state.exp_avg();
        auto& v = state.exp_avg_sq();
        double beta1 = group_options.beta1();
        double beta2 = group_options.beta2();

        // Apply decoupled weight decay (AdamW style)
        if (group_options.weight_decay() > 0.0) {
            p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
        }

        // 1. Update biased moments
        m.mul_(beta1).add_(grad, 1.0 - beta1);
        v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

        // 2. NAdam's key modification to the momentum update
        // Bias correction for m_t and v_t
        double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
        double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

        auto m_hat = m / bias_correction1;

        // This is the "lookahead" part. It combines the bias-corrected momentum
        // with the bias-corrected version of the current raw gradient.
        auto grad_hat = grad / bias_correction1;
        auto m_hat_prime = m_hat * beta1 + grad_hat * (1.0 - beta1);

        // 3. Denominator (same as Adam)
        auto v_hat = v / bias_correction2;
        auto denom = v_hat.sqrt().add(group_options.eps());

        // 4. Final Update
        p.data().addcdiv_(m_hat_prime, denom, -group_options.lr());
    }
    return loss;
}

// --- Boilerplate Methods ---
void NAdam::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void NAdam::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> NAdam::make_param_state() { return std::make_unique<NAdamParamState>(); }