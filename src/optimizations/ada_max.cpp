#include "include/optimizations/ada_max.h"
#include <stdexcept>
namespace xt::optim
{
    // --- AdaMaxOptions Methods ---
    void AdaMaxOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }
    void AdaMaxOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AdaMaxOptions::clone() const {
        auto cloned = std::make_unique<AdaMaxOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- AdaMaxParamState Methods ---
    void AdaMaxParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_inf_norm().defined()) archive.write("exp_inf_norm", exp_inf_norm(), true);
    }
    void AdaMaxParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_inf_norm", temp, true)) { exp_inf_norm_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaMaxParamState::clone() const {
        auto cloned = std::make_unique<AdaMaxParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_inf_norm().defined()) cloned->exp_inf_norm(exp_inf_norm().clone());
        return cloned;
    }


    // --- AdaMax Implementation ---
    AdaMax::AdaMax(std::vector<torch::Tensor> params, AdaMaxOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AdaMaxOptions>(options)) {}

    AdaMax::AdaMax(std::vector<torch::Tensor> params, double lr)
        : AdaMax(std::move(params), AdaMaxOptions(lr)) {}

    torch::Tensor AdaMax::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AdaMaxOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AdaMax optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<AdaMaxParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                    state.exp_inf_norm(torch::zeros_like(p));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();

                // Apply classic L2 regularization (weight decay)
                if (group_options.weight_decay() > 0.0) {
                    grad = grad.add(p.detach(), group_options.weight_decay());
                }

                auto& m = state.exp_avg();
                auto& u = state.exp_inf_norm();
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();

                // 1. Update first moment (m_t) - same as Adam
                m.mul_(beta1).add_(grad, 1.0 - beta1);

                // 2. Update exponentially weighted infinity norm (u_t)
                // u_t = max(beta2 * u_{t-1}, |g_t|)
                auto u_prev_scaled = u * beta2;
                torch::max_out(u, u_prev_scaled, grad.abs()); // u = max(u_prev_scaled, |grad|)

                // 3. Compute bias-corrected first moment
                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                auto m_hat = m / bias_correction1;

                // 4. Final update
                // Denominator is just u_t + eps. No sqrt, no bias correction for u_t.
                auto denom = u + group_options.eps();

                p.data().addcdiv_(m_hat, denom, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AdaMax::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AdaMax::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaMax::make_param_state() { return std::make_unique<AdaMaxParamState>(); }
}