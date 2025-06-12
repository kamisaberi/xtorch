#include "include/optimizations/gravity.h"
#include <stdexcept>

// --- GravityOptions Methods ---
void GravityOptions::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("lr", this->lr());
    archive.write("beta1", beta1());
    archive.write("beta2", beta2());
    archive.write("eps", eps());
    archive.write("gravitational_constant", gravitational_constant());
}

void GravityOptions::deserialize(torch::serialize::InputArchive& archive) {
    c10::IValue ivalue;
    if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
    if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
    if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
    if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    if (archive.try_read("gravitational_constant", ivalue)) { gravitational_constant_ = ivalue.toDouble(); }
}

std::unique_ptr<torch::optim::OptimizerOptions> GravityOptions::clone() const {
    auto cloned = std::make_unique<GravityOptions>(this->lr());
    cloned->beta1(beta1()).beta2(beta2()).eps(eps())
          .gravitational_constant(gravitational_constant());
    return cloned;
}

// --- GravityParamState Methods ---
void GravityParamState::serialize(torch::serialize::OutputArchive& archive) const {
    archive.write("step", step(), true);
    if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
}

void GravityParamState::deserialize(torch::serialize::InputArchive& archive) {
    at::Tensor temp;
    if(archive.try_read("step", temp, true)) { step_ = temp; }
    if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
}

std::unique_ptr<torch::optim::OptimizerParamState> GravityParamState::clone() const {
    auto cloned = std::make_unique<GravityParamState>();
    if(step().defined()) cloned->step(step().clone());
    if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
    if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
    return cloned;
}

// --- Gravity Optimizer Implementation ---
Gravity::Gravity(std::vector<torch::Tensor> params, GravityOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<GravityOptions>(options)) {}

torch::Tensor Gravity::step(LossClosure closure) {
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    auto& group_options = static_cast<GravityOptions&>(param_groups_[0].options());

    for (auto& p : param_groups_[0].params()) {
        if (!p.grad().defined()) { continue; }

        auto grad = p.grad();
        auto& state = static_cast<GravityParamState&>(*state_.at(p.unsafeGetTensorImpl()));

        if (!state.step().defined()) {
            state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
            state.exp_avg(torch::zeros_like(p));
            state.exp_avg_sq(torch::zeros_like(p));
        }
        state.step(state.step() + 1.0);
        double current_step_val = state.step().item<double>();

        // 1. Calculate the "Gravitational" regularization force.
        // This force pulls the parameter towards zero, proportional to its own magnitude ("mass").
        // F_gravity = G * m1 * m2 / r^2 --> Simplified to: F_gravity = k * param
        auto gravitational_force = p.detach() * group_options.gravitational_constant();

        // 2. The total force is the sum of the gradient and the gravitational pull.
        auto total_force = grad + gravitational_force;

        // 3. Use Adam's mechanism to process this total force.
        auto& m = state.exp_avg();
        auto& v = state.exp_avg_sq();
        double beta1 = group_options.beta1();
        double beta2 = group_options.beta2();

        // Update momentum (m_t) on the total force
        m.mul_(beta1).add_(total_force, 1.0 - beta1);

        // Update curvature estimate (v_t) based on the gradient *only*.
        // The curvature of the loss landscape is independent of our regularization force.
        // This is a key design choice, similar to how AdamW decouples weight decay.
        v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

        // 4. Bias correction
        double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
        double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

        auto m_hat = m / bias_correction1;
        auto v_hat = v / bias_correction2;

        // 5. Final update
        // The denominator adapts the step size based on the loss curvature.
        auto denom = v_hat.sqrt().add_(group_options.eps());

        // The update is driven by the momentum of the total force.
        p.data().addcdiv_(m_hat, denom, -group_options.lr());
    }
    return loss;
}

// --- Boilerplate Methods ---
void Gravity::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
void Gravity::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
std::unique_ptr<torch::optim::OptimizerParamState> Gravity::make_param_state() { return std::make_unique<GravityParamState>(); }