#include <optimizations/r_adam.h>
#include <stdexcept>

namespace xt::optim
{
    // --- RAdamOptions Methods ---
    void RAdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }
    void RAdamOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> RAdamOptions::clone() const {
        auto cloned = std::make_unique<RAdamOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- RAdamParamState Methods ---
    void RAdamParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }
    void RAdamParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> RAdamParamState::clone() const {
        auto cloned = std::make_unique<RAdamParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }


    // --- RAdam Implementation ---
    RAdam::RAdam(std::vector<torch::Tensor> params, RAdamOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<RAdamOptions>(options)) {}

    RAdam::RAdam(std::vector<torch::Tensor> params, double lr)
        : RAdam(std::move(params), RAdamOptions(lr)) {}

    torch::Tensor RAdam::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<RAdamOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("RAdam optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<RAdamParamState&>(*state_.at(p.unsafeGetTensorImpl()));

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

                // Apply decoupled weight decay
                if (group_options.weight_decay() > 0.0) {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                // Update biased moments
                m.mul_(beta1).add_(grad, 1.0 - beta1);
                v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                // Bias corrected first moment
                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                auto m_hat = m / bias_correction1;

                // --- RAdam Rectification Logic ---
                // Calculate rho_inf and rho_t
                double rho_inf = 2.0 / (1.0 - beta2) - 1.0;
                double rho_t = rho_inf - 2.0 * current_step_val * std::pow(beta2, current_step_val) / (1.0 - std::pow(beta2, current_step_val));

                // Check if rectification is needed
                if (rho_t > 5.0) {
                    // Variance is reliable, perform rectified Adam update
                    // Calculate rectification term r_t
                    double r_t_num = (rho_t - 4.0) * (rho_t - 2.0) * rho_inf;
                    double r_t_den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho_t;
                    double r_t = std::sqrt(r_t_num / r_t_den);

                    // Denominator is rectified
                    double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);
                    auto v_hat = v / bias_correction2;
                    auto denom = v_hat.sqrt().add(group_options.eps());

                    // Final update with rectification term r_t
                    p.data().addcdiv_(m_hat, denom, -group_options.lr() * r_t);
                } else {
                    // Variance is not yet reliable, perform SGD with momentum update
                    // (i.e., turn off the adaptive part)
                    p.data().add_(m_hat, -group_options.lr());
                }
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void RAdam::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void RAdam::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> RAdam::make_param_state() { return std::make_unique<RAdamParamState>(); }
}