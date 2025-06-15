#include "include/optimizations/ada_bound.h"
#include <stdexcept>

// --- AdaBoundOptions Methods (Correct) ---
void AdaBoundOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta1", beta1()); archive.write("beta2", beta2());
    archive.write("final_lr", final_lr()); archive.write("gamma", gamma());
    archive.write("eps", eps()); archive.write("weight_decay", weight_decay());
}
void AdaBoundOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("final_lr", ivalue)) { final_lr_ = ivalue.toDouble(); }
    if (archive.try_read("gamma", ivalue)) { gamma_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> AdaBoundOptions::clone() const {
    auto cloned = std::make_unique<AdaBoundOptions>(this->lr());
    cloned->beta1(beta1()).beta2(beta2()).final_lr(final_lr()).gamma(gamma())
          .eps(eps()).weight_decay(weight_decay());
    return cloned;
}

// --- AdaBoundParamState Methods (Correct, Serialization handled by TORCH_ARG) ---
// Note: We don't need to write serialize/deserialize for AdaBoundParamState
// because TORCH_ARG handles it automatically. The clone() method is still good practice.

// --- AdaBound Implementation (Correct) ---
AdaBound::AdaBound(std::vector<torch::Tensor> params, AdaBoundOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<AdaBoundOptions>(options)) {}

AdaBound::AdaBound(std::vector<torch::Tensor> params, double lr)
    : AdaBound(std::move(params), AdaBoundOptions(lr)) {}

torch::Tensor AdaBound::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    for (auto& group : param_groups_) {
        auto& options = static_cast<AdaBoundOptions&>(group.options());

        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }
            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("AdaBound optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<AdaBoundParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            if (options.weight_decay() > 0.0) {
                grad = grad.add(p.detach(), options.weight_decay());
            }

            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            m.mul_(options.beta1()).add_(grad, 1.0 - options.beta1());
            v.mul_(options.beta2()).addcmul_(grad, grad, 1.0 - options.beta2());

            double bias_correction1 = 1.0 - std::pow(options.beta1(), current_step_val);
            double bias_correction2 = 1.0 - std::pow(options.beta2(), current_step_val);
            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;

            auto denom = v_hat.sqrt().add_(options.eps());

            double final_lr = options.final_lr();
            double lower_bound = final_lr * (1.0 - 1.0 / (options.gamma() * current_step_val + 1.0));
            double upper_bound = final_lr * (1.0 + 1.0 / (options.gamma() * current_step_val + 1.0));

            auto effective_lr_base = options.lr() / denom;
            auto final_lr_per_param = torch::clamp(effective_lr_base, lower_bound, upper_bound);

            p.data().addcmul_(m_hat, final_lr_per_param, -1.0);
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
std::unique_ptr<torch::optim::OptimizerParamState> AdaBound::make_param_state() {
    return std::make_unique<AdaBoundParamState>();
}

void AdaBound::save(torch::serialize::OutputArchive& archive) const {
    torch::optim::Optimizer::save(archive);
}

void AdaBound::load(torch::serialize::InputArchive& archive) {
    torch::optim::Optimizer::load(archive);
}