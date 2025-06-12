#include "include/optimizations/forward_gradient.h"
#include <stdexcept>

// --- ForwardGradientOptions Methods ---
void ForwardGradientOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr);
    archive.write("alpha", alpha());
    archive.write("k", k());
    archive.write("beta1", beta1());
    archive.write("beta2", beta2());
    archive.write("eps", eps());
    archive.write("weight_decay", weight_decay());
}

void ForwardGradientOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("alpha", ivalue)) { alpha_ = ivalue.toDouble(); }
    if (archive.try_read("k", ivalue)) { k_ = ivalue.toInt(); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
}

std::unique_ptr<torch::optim::OptimizerOptions> ForwardGradientOptions::clone() const {
    auto cloned = std::make_unique<ForwardGradientOptions>(this->lr);
    cloned->alpha(alpha()).k(k()).beta1(beta1()).beta2(beta2())
          .eps(eps()).weight_decay(weight_decay());
    return cloned;
}

// --- ForwardGradientParamState Methods ---
void ForwardGradientParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(slow_param().defined()) archive.write("slow_param", slow_param(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
}

void ForwardGradientParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("slow_param", temp, true)) { slow_param_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
}

std::unique_ptr<torch::optim::OptimizerParamState> ForwardGradientParamState::clone() const {
    auto cloned = std::make_unique<ForwardGradientParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(slow_param().defined()) cloned->slow_param(slow_param().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    return cloned;
}

// --- ForwardGradient (Lookahead) Implementation ---
ForwardGradient::ForwardGradient(std::vector<torch::Tensor> params, ForwardGradientOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<ForwardGradientOptions>(options)) {
    // Initialize slow parameters
    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            auto& state = static_cast<ForwardGradientParamState&>(*state_.at(p.unsafeGetTensorImpl()));
            state.slow_param(p.detach().clone());
        }
    }
}

torch::Tensor ForwardGradient::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<ForwardGradientOptions&>(param_groups_[0].options());

    for (auto& p : param_groups_[0].params()) {
        if (!p.grad().defined()) { continue; }

        auto grad = p.grad();
        if (grad.is_sparse()) {
            throw std::runtime_error("ForwardGradient (Lookahead) does not support sparse gradients.");
        }

        auto& state = static_cast<ForwardGradientParamState&>(*state_.at(p.unsafeGetTensorImpl()));

        if (!state.step().defined()) {
            state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
            state.exp_avg(torch::zeros_like(p));
            state.exp_avg_sq(torch::zeros_like(p));
            // slow_param is initialized in constructor
        }
        state.step(state.step() + 1.0);
        double current_step_val = state.step().item<double>();

        // --- Inner Adam Optimizer Step (updates the "fast weights", which are p.data()) ---
        auto& m = state.exp_avg();
        auto& v = state.exp_avg_sq();
        double bias_correction1 = 1.0 - std::pow(group_options.beta1(), current_step_val);
        double bias_correction2 = 1.0 - std::pow(group_options.beta2(), current_step_val);

        if (group_options.weight_decay() > 0.0) {
            grad.add_(p.detach(), group_options.weight_decay());
        }

        m.mul_(group_options.beta1()).add_(grad, 1.0 - group_options.beta1());
        v.mul_(group_options.beta2()).addcmul_(grad, grad, 1.0 - group_options.beta2());

        auto m_hat = m / bias_correction1;
        auto v_hat = v / bias_correction2;
        auto denom = v_hat.sqrt().add_(group_options.eps());

        // This updates the fast weights (p)
        p.data().addcdiv_(m_hat, denom, -group_options.lr);

        // --- Lookahead Synchronization Step ---
        if (static_cast<long>(current_step_val) % group_options.k() == 0) {
            auto& slow_p = state.slow_param();
            // Update slow weights: slow_p += alpha * (fast_p - slow_p)
            slow_p.add_(p.detach() - slow_p, group_options.alpha());
            // Sync fast weights back to slow weights
            p.data().copy_(slow_p);
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void ForwardGradient::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void ForwardGradient::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> ForwardGradient::make_param_state() { return std::make_unique<ForwardGradientParamState>(); }