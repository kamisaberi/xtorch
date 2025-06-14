#include "include/optimizations/ada_mod.h"
#include <stdexcept>

// --- AdaModOptions Methods ---
void AdaModOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta1", beta1());
    archive.write("beta2", beta2());
    archive.write("beta3", beta3());
    archive.write("eps", eps());
    archive.write("weight_decay", weight_decay());
}
void AdaModOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("beta3", ivalue)) { beta3_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> AdaModOptions::clone() const {
    auto cloned = std::make_unique<AdaModOptions>(this->lr());
    cloned->beta1(beta1()).beta2(beta2()).beta3(beta3()).eps(eps()).weight_decay(weight_decay());
    return cloned;
}

// --- AdaModParamState Methods ---
void AdaModParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    if(long_term_avg_sq().defined()) archive.write("long_term_avg_sq", long_term_avg_sq(), true);
}
void AdaModParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    if(archive.try_read("long_term_avg_sq", temp, true)) { long_term_avg_sq_ = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> AdaModParamState::clone() const {
    auto cloned = std::make_unique<AdaModParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    if(long_term_avg_sq().defined()) cloned->long_term_avg_sq(long_term_avg_sq().clone());
    return cloned;
}


// --- AdaMod Implementation ---
AdaMod::AdaMod(std::vector<torch::Tensor> params, AdaModOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<AdaModOptions>(options)) {}

AdaMod::AdaMod(std::vector<torch::Tensor> params, double lr)
    : AdaMod(std::move(params), AdaModOptions(lr)) {}

torch::Tensor AdaMod::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<AdaModOptions&>(param_groups_[0].options());

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("AdaMod optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<AdaModParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
                state.long_term_avg_sq(torch::zeros_like(p)); // Initialize s_t
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            // Apply decoupled weight decay
            if (group_options.weight_decay() > 0.0) {
                p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
            }

            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            auto& s = state.long_term_avg_sq();
            double beta1 = group_options.beta1();
            double beta2 = group_options.beta2();

            // 1. Standard Adam moment updates
            m.mul_(beta1).add_(grad, 1.0 - beta1);
            v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

            double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
            double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;

            // 2. AdaMod's long-term memory update
            // s_t = max(s_{t-1}, sqrt(v_hat_t))
            // The paper also proposes an EMA variant: s_t = beta3*s_{t-1} + (1-beta3)*sqrt(v_hat)
            // Let's implement the EMA variant as it's more general.
            auto current_v_hat_sqrt = v_hat.sqrt();
            s.mul_(group_options.beta3()).add_(current_v_hat_sqrt, 1.0 - group_options.beta3());

            // 3. Final update using the long-term memory as the denominator
            auto denom = s + group_options.eps();

            p.data().addcdiv_(m_hat, denom, -group_options.lr());
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void AdaMod::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void AdaMod::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> AdaMod::make_param_state() { return std::make_unique<AdaModParamState>(); }