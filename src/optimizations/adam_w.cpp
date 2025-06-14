#include "include/optimizations/adam_w.h"
#include <stdexcept>

// --- AdamWOptions Methods ---
void AdamWOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta1", beta1());
    archive.write("beta2", beta2());
    archive.write("eps", eps());
    archive.write("weight_decay", weight_decay());
    archive.write("amsgrad", amsgrad());
}
void AdamWOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    if (archive.try_read("amsgrad", ivalue)) { amsgrad_ = ivalue.toBool(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> AdamWOptions::clone() const {
    auto cloned = std::make_unique<AdamWOptions>(this->lr());
    cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay()).amsgrad(amsgrad());
    return cloned;
}

// --- AdamWParamState Methods ---
void AdamWParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    if(max_exp_avg_sq().defined()) archive.write("max_exp_avg_sq", max_exp_avg_sq(), true);
}
void AdamWParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    if(archive.try_read("max_exp_avg_sq", temp, true)) { max_exp_avg_sq_ = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> AdamWParamState::clone() const {
    auto cloned = std::make_unique<AdamWParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    if(max_exp_avg_sq().defined()) cloned->max_exp_avg_sq(max_exp_avg_sq().clone());
    return cloned;
}


// --- AdamW Implementation ---
AdamW::AdamW(std::vector<torch::Tensor> params, AdamWOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<AdamWOptions>(options)) {}

AdamW::AdamW(std::vector<torch::Tensor> params, double lr)
    : AdamW(std::move(params), AdamWOptions(lr)) {}

torch::Tensor AdamW::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    for (auto& group : param_groups_) {
        auto& options = static_cast<AdamWOptions&>(group.options());
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("AdamW optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<AdamWParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
                if (options.amsgrad()) {
                    state.max_exp_avg_sq(torch::zeros_like(p));
                }
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            double beta1 = options.beta1();
            double beta2 = options.beta2();

            // 1. Decoupled Weight Decay
            // This is the key difference from standard Adam.
            // p = p - lr * wd * p
            if (options.weight_decay() > 0.0) {
                p.data().mul_(1.0 - options.lr() * options.weight_decay());
            }

            // 2. Standard Adam moment updates (using the raw gradient)
            m.mul_(beta1).add_(grad, 1.0 - beta1);
            v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

            // 3. Bias correction
            double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
            auto m_hat = m / bias_correction1;

            torch::Tensor denom;
            if (options.amsgrad()) {
                auto& max_v = state.max_exp_avg_sq();
                torch::max_out(max_v, max_v, v); // max_v = max(max_v, v)
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);
                auto v_hat = max_v / bias_correction2;
                denom = v_hat.sqrt().add_(options.eps());
            } else {
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);
                auto v_hat = v / bias_correction2;
                denom = v_hat.sqrt().add_(options.eps());
            }

            // 4. Final update
            p.data().addcdiv_(m_hat, denom, -options.lr());
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void AdamW::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void AdamW::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> AdamW::make_param_state() { return std::make_unique<AdamWParamState>(); }