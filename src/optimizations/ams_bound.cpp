#include "include/optimizations/ams_bound.h"
#include <stdexcept>

// --- AMSBoundOptions Methods ---
void AMSBoundOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta1", beta1()); archive.write("beta2", beta2());
    archive.write("final_lr", final_lr()); archive.write("gamma", gamma());
    archive.write("eps", eps()); archive.write("weight_decay", weight_decay());
}
void AMSBoundOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("final_lr", ivalue)) { final_lr_ = ivalue.toDouble(); }
    if (archive.try_read("gamma", ivalue)) { gamma_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
}
std::unique_ptr<torch::optim::OptimizerOptions> AMSBoundOptions::clone() const {
    auto cloned = std::make_unique<AMSBoundOptions>(this->lr());
    cloned->beta1(beta1()).beta2(beta2()).final_lr(final_lr()).gamma(gamma())
          .eps(eps()).weight_decay(weight_decay());
    return cloned;
}

// --- AMSBoundParamState Methods ---
void AMSBoundParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    if(max_exp_avg_sq().defined()) archive.write("max_exp_avg_sq", max_exp_avg_sq(), true);
}
void AMSBoundParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    if(archive.try_read("max_exp_avg_sq", temp, true)) { max_exp_avg_sq_ = temp; }
}
std::unique_ptr<torch::optim::OptimizerParamState> AMSBoundParamState::clone() const {
    auto cloned = std::make_unique<AMSBoundParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    if(max_exp_avg_sq().defined()) cloned->max_exp_avg_sq(max_exp_avg_sq().clone());
    return cloned;
}


// --- AMSBound Implementation ---
AMSBound::AMSBound(std::vector<torch::Tensor> params, AMSBoundOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<AMSBoundOptions>(options)) {}

AMSBound::AMSBound(std::vector<torch::Tensor> params, double lr)
    : AMSBound(std::move(params), AMSBoundOptions(lr)) {}

torch::Tensor AMSBound::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<AMSBoundOptions&>(param_groups_[0].options());

    for (auto& group : param_groups_) {
        for (auto& p : group.params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("AMSBound optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<AMSBoundParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
                state.max_exp_avg_sq(torch::zeros_like(p)); // For AMSGrad
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            // Decoupled weight decay
            if (group_options.weight_decay() > 0.0) {
                p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
            }

            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            auto& max_v = state.max_exp_avg_sq();

            // 1. Standard moment updates
            m.mul_(group_options.beta1()).add_(grad, 1.0 - group_options.beta1());
            v.mul_(group_options.beta2()).addcmul_(grad, grad, 1.0 - group_options.beta2());

            // 2. AMSGrad part: use the max of v_t seen so far
            torch::max_out(max_v, max_v, v);

            double bias_correction1 = 1.0 - std::pow(group_options.beta1(), current_step_val);
            double bias_correction2 = 1.0 - std::pow(group_options.beta2(), current_step_val);

            // Denominator is based on the AMSGrad state
            auto denom = (max_v / bias_correction2).sqrt().add_(group_options.eps());

            // 3. Bounding part: calculate dynamic bounds
            // The bounds converge to the final_lr as t -> infinity
            double lower_bound = group_options.final_lr() * (1.0 - 1.0 / (group_options.gamma() * current_step_val + 1.0));
            double upper_bound = group_options.final_lr() * (1.0 + 1.0 / (group_options.gamma() * current_step_val));

            // The effective per-parameter learning rate from Adam/AMSGrad
            auto effective_lr_base = group_options.lr() / denom;

            // Clip this effective learning rate between the bounds
            auto final_lr = torch::clamp(effective_lr_base, lower_bound, upper_bound);

            // 4. Final update
            // update = final_lr * (m / bias_correction1)
            auto m_hat = m / bias_correction1;
            p.data().addcmul_(m_hat, final_lr, -1.0);
        }
    }
    return loss;
}

// --- Boilerplate Methods ---
void AMSBound::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void AMSBound::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> AMSBound::make_param_state() { return std::make_unique<AMSBoundParamState>(); }