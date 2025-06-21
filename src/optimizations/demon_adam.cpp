#include "include/optimizations/demon_adam.h"
#include <stdexcept>
#include <algorithm> // For std::min
namespace xt::optim
{
    // --- DemonAdamOptions Methods ---
    void DemonAdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1_initial", beta1_initial());
        archive.write("beta1_final", beta1_final());
        archive.write("total_steps", total_steps());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }
    void DemonAdamOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1_initial", ivalue)) { beta1_initial_ = ivalue.toDouble(); }
        if (archive.try_read("beta1_final", ivalue)) { beta1_final_ = ivalue.toDouble(); }
        if (archive.try_read("total_steps", ivalue)) { total_steps_ = ivalue.toInt(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> DemonAdamOptions::clone() const {
        auto cloned = std::make_unique<DemonAdamOptions>(this->lr());
        cloned->beta1_initial(beta1_initial()).beta1_final(beta1_final())
              .total_steps(total_steps()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- DemonAdamParamState Methods ---
    void DemonAdamParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }
    void DemonAdamParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> DemonAdamParamState::clone() const {
        auto cloned = std::make_unique<DemonAdamParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }

    // --- DemonAdam Implementation ---
    DemonAdam::DemonAdam(std::vector<torch::Tensor> params, DemonAdamOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<DemonAdamOptions>(options)) {}

    DemonAdam::DemonAdam(std::vector<torch::Tensor> params, double lr)
        : DemonAdam(std::move(params), DemonAdamOptions(lr)) {}

    torch::Tensor DemonAdam::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<DemonAdamOptions&>(param_groups_[0].options());

        // Get the global step from the first parameter to calculate beta1_t
        long global_step = 0;
        if (!param_groups_.empty() && !param_groups_[0].params().empty()) {
            auto& p_ref = param_groups_[0].params()[0];
            auto& state_ref = static_cast<DemonAdamParamState&>(*state_.at(p_ref.unsafeGetTensorImpl()));
            if (state_ref.step().defined()) {
                global_step = static_cast<long>(state_ref.step().item<double>());
            }
        }
        global_step++;

        // 1. Calculate the dynamic beta1 for this step
        double decay_progress = std::min(1.0, static_cast<double>(global_step) / group_options.total_steps());
        double beta1_t = group_options.beta1_initial() -
                         (group_options.beta1_initial() - group_options.beta1_final()) * decay_progress;

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("DemonAdam optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<DemonAdamParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                    state.exp_avg_sq(torch::zeros_like(p));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>(); // Use local step for bias correction

                // Apply decoupled weight decay (AdamW style)
                if (group_options.weight_decay() > 0.0) {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                auto& m = state.exp_avg();
                auto& v = state.exp_avg_sq();
                double beta2 = group_options.beta2();

                // 2. Update Adam moments, using the dynamic beta1_t
                m.mul_(beta1_t).add_(grad, 1.0 - beta1_t);
                v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                // 3. Bias correction
                // Use the dynamic beta1_t for the first moment's bias correction
                double bias_correction1 = 1.0 - std::pow(beta1_t, current_step_val);
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

                auto m_hat = m / bias_correction1;
                auto v_hat = v / bias_correction2;

                // 4. Final Adam update
                auto denom = v_hat.sqrt().add(group_options.eps());
                p.data().addcdiv_(m_hat, denom, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void DemonAdam::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void DemonAdam::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> DemonAdam::make_param_state() { return std::make_unique<DemonAdamParamState>(); }
}