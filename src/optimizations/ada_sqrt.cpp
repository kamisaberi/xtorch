#include "include/optimizations/ada_sqrt.h"
#include <stdexcept>

namespace xt::optim
{
    // --- AdaSqrtOptions Methods ---
    void AdaSqrtOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }
    void AdaSqrtOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AdaSqrtOptions::clone() const {
        auto cloned = std::make_unique<AdaSqrtOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- AdaSqrtParamState Methods ---
    void AdaSqrtParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_abs().defined()) archive.write("exp_avg_abs", exp_avg_abs(), true);
    }
    void AdaSqrtParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_abs", temp, true)) { exp_avg_abs_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaSqrtParamState::clone() const {
        auto cloned = std::make_unique<AdaSqrtParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_abs().defined()) cloned->exp_avg_abs(exp_avg_abs().clone());
        return cloned;
    }

    // --- AdaSqrt Implementation ---
    AdaSqrt::AdaSqrt(std::vector<torch::Tensor> params, AdaSqrtOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AdaSqrtOptions>(options)) {}

    AdaSqrt::AdaSqrt(std::vector<torch::Tensor> params, double lr)
        : AdaSqrt(std::move(params), AdaSqrtOptions(lr)) {}

    torch::Tensor AdaSqrt::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AdaSqrtOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AdaSqrt optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<AdaSqrtParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                    state.exp_avg_abs(torch::zeros_like(p));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();

                // Apply decoupled weight decay
                if (group_options.weight_decay() > 0.0) {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                auto& m = state.exp_avg();
                auto& v = state.exp_avg_abs();
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();

                // 1. Update first moment (m_t) - standard momentum
                m.mul_(beta1).add_(grad, 1.0 - beta1);

                // 2. Update second moment (v_t) with EMA of absolute gradients
                v.mul_(beta2).add_(grad.abs(), 1.0 - beta2);

                // 3. Bias correction for both moments
                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

                auto m_hat = m / bias_correction1;
                auto v_hat = v / bias_correction2;

                // 4. Final update using sqrt of v_hat as the denominator
                auto denom = v_hat.sqrt().add(group_options.eps());

                p.data().addcdiv_(m_hat, denom, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AdaSqrt::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AdaSqrt::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaSqrt::make_param_state() { return std::make_unique<AdaSqrtParamState>(); }
}