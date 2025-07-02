#include "include/optimizations/ao.h"
#include <stdexcept>
namespace xt::optim
{
    // --- AOOptions Methods ---
    void AOOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1()); archive.write("beta2", beta2());
        archive.write("eps", eps()); archive.write("weight_decay", weight_decay());
        archive.write("overdrive_strength", overdrive_strength());
    }
    void AOOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("overdrive_strength", ivalue)) { overdrive_strength_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AOOptions::clone() const {
        auto cloned = std::make_unique<AOOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps())
              .weight_decay(weight_decay()).overdrive_strength(overdrive_strength());
        return cloned;
    }

    // --- AOParamState Methods ---
    void AOParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }
    void AOParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AOParamState::clone() const {
        auto cloned = std::make_unique<AOParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }

    // --- AO Implementation ---
    AO::AO(std::vector<torch::Tensor> params, AOOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AOOptions>(options)) {}

    AO::AO(std::vector<torch::Tensor> params, double lr)
        : AO(std::move(params), AOOptions(lr)) {}

    torch::Tensor AO::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AOOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AO optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<AOParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                    state.exp_avg_sq(torch::zeros_like(p));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();

                // Apply decoupled weight decay
                if (group_options.weight_decay() > 0.0) {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                auto& m = state.exp_avg();
                auto& v = state.exp_avg_sq();
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();

                // 1. Calculate the Overdrive Factor (omega)
                double overdrive_factor = 1.0;
                if (current_step_val > 1) { // Need history for momentum
                    auto grad_norm = grad.norm().item<double>();
                    auto m_norm = m.norm().item<double>();

                    if (grad_norm > group_options.eps() && m_norm > group_options.eps()) {
                        auto dot_product = (grad * m).sum().item<double>();
                        auto cosine_similarity = dot_product / (grad_norm * m_norm);

                        // Overdrive is 1 + (strength * similarity), clamped to be >= 1
                        overdrive_factor = 1.0 + group_options.overdrive_strength() * cosine_similarity;
                        overdrive_factor = std::max(1.0, overdrive_factor);
                    }
                }

                // 2. Standard moment updates
                m.mul_(beta1).add_(grad, 1.0 - beta1);
                v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                // 3. Bias correction
                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

                // Apply overdrive to the bias-corrected momentum
                auto m_hat = (m / bias_correction1) * overdrive_factor;
                auto v_hat = v / bias_correction2;

                // 4. Final Adam-like update
                auto denom = v_hat.sqrt().add_(group_options.eps());
                p.data().addcdiv_(m_hat, denom, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AO::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AO::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AO::make_param_state() { return std::make_unique<AOParamState>(); }
}