#include "include/optimizations/power_propagation.h"
#include <stdexcept>

namespace xt::optim
{
    // --- PowerPropOptions Methods ---
    void PowerPropOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
        archive.write("power", power());
    }
    void PowerPropOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("power", ivalue)) { power_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> PowerPropOptions::clone() const {
        auto cloned = std::make_unique<PowerPropOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay()).power(power());
        return cloned;
    }

    // --- PowerPropParamState Methods ---
    void PowerPropParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }
    void PowerPropParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> PowerPropParamState::clone() const {
        auto cloned = std::make_unique<PowerPropParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }

    // --- PowerPropagation Implementation ---
    PowerPropagation::PowerPropagation(std::vector<torch::Tensor> params, PowerPropOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<PowerPropOptions>(options)) {}

    PowerPropagation::PowerPropagation(std::vector<torch::Tensor> params, double lr)
        : PowerPropagation(std::move(params), PowerPropOptions(lr)) {}


    torch::Tensor PowerPropagation::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<PowerPropOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("PowerPropagation optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<PowerPropParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            // Apply decoupled weight decay directly to parameters
            if (group_options.weight_decay() > 0.0) {
                p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
            }

            // 1. Apply the PowerPropagation transformation to the gradient
            // powered_grad = sign(grad) * |grad|^p
            auto grad_sign = torch::sign(grad);
            auto grad_abs_powered = torch::pow(torch::abs(grad), group_options.power());
            auto powered_grad = grad_sign * grad_abs_powered;

            // 2. Feed the transformed gradient into a standard Adam update
            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            double beta1 = group_options.beta1();
            double beta2 = group_options.beta2();

            m.mul_(beta1).add_(powered_grad, 1.0 - beta1);
            v.mul_(beta2).addcmul_(powered_grad, powered_grad, 1.0 - beta2);

            // 3. Bias correction
            double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
            double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;

            // 4. Final update
            auto denom = v_hat.sqrt().add_(group_options.eps());
            p.data().addcdiv_(m_hat, denom, -group_options.lr());
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void PowerPropagation::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void PowerPropagation::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> PowerPropagation::make_param_state() { return std::make_unique<PowerPropParamState>(); }
}