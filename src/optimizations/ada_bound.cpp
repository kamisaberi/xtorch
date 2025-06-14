#include "include/optimizations/ada_bound.h"
#include <stdexcept>

// --- AdaBoundOptions Methods ---
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

// --- AdaBoundParamState Methods ---
void AdaBoundParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
}
void AdaBoundParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> AdaBoundParamState::clone() const {
    auto cloned = std::make_unique<AdaBoundParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    return cloned;
}


// --- AdaBound Implementation ---
AdaBound::AdaBound(std::vector<torch::Tensor> params, AdaBoundOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<AdaBoundOptions>(options)) {}

AdaBound::AdaBound(std::vector<torch::Tensor> params, double lr)
    : AdaBound(std::move(params), AdaBoundOptions(lr)) {}

torch::Tensor AdaBound::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<AdaBoundOptions&>(param_groups_[0].options());

    for (auto& group : param_groups_) {
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

            // Apply classic L2 regularization
            if (group_options.weight_decay() > 0.0) {
                grad = grad.add(p.detach(), group_options.weight_decay());
            }

            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            double beta1 = group_options.beta1();
            double beta2 = group_options.beta2();

            // 1. Standard Adam moment updates
            m.mul_(beta1).add_(grad, 1.0 - beta1);
            v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

            // 2. Bias correction
            double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
            double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);
            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;

            // This is the denominator from the standard Adam update
            auto denom = v_hat.sqrt().add_(group_options.eps());

            // 3. Compute dynamic bounds for this step
            double final_lr = group_options.final_lr() * group_options.lr() / this->defaults().lr();
            double lower_bound = final_lr * (1.0 - 1.0 / (group_options.gamma() * current_step_val + 1.0));
            double upper_bound = final_lr * (1.0 + 1.0 / (group_options.gamma() * current_step_val + 1.0));

            // 4. Clip the base learning rate and then perform the update
            // AdaBound clips the final step size, not just the LR.
            // step_size = clip(lr / denom, lower_bound, upper_bound)
            // The paper's formulation is equivalent to clipping lr/denom.

            // Effective per-parameter LR from Adam
            auto effective_lr_base = group_options.lr() / denom;

            // Clip this effective learning rate tensor element-wise
            auto final_lr_per_param = torch::clamp(effective_lr_base, lower_bound, upper_bound);

            // 5. Final update
            // p_t = p_{t-1} - clipped_lr_per_param * m_hat
            p.data().addcmul_(m_hat, final_lr_per_param, -1.0);
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void AdaBound::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void AdaBound::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> AdaBound::make_param_state() { return std::make_unique<AdaBoundParamState>(); }