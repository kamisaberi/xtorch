#include "include/optimizations/ada_fisher.h"
#include <stdexcept>
namespace xt::optim
{
    // --- AdaFisherOptions Methods ---
    void AdaFisherOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }
    void AdaFisherOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AdaFisherOptions::clone() const {
        auto cloned = std::make_unique<AdaFisherOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- AdaFisherParamState Methods ---
    void AdaFisherParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(fisher_diag_ema().defined()) archive.write("fisher_diag_ema", fisher_diag_ema(), true);
    }
    void AdaFisherParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("fisher_diag_ema", temp, true)) { fisher_diag_ema_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaFisherParamState::clone() const {
        auto cloned = std::make_unique<AdaFisherParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(fisher_diag_ema().defined()) cloned->fisher_diag_ema(fisher_diag_ema().clone());
        return cloned;
    }

    // --- AdaFisher Implementation ---
    AdaFisher::AdaFisher(std::vector<torch::Tensor> params, AdaFisherOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AdaFisherOptions>(options)) {}

    AdaFisher::AdaFisher(std::vector<torch::Tensor> params, double lr)
        : AdaFisher(std::move(params), AdaFisherOptions(lr)) {}

    torch::Tensor AdaFisher::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AdaFisherOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AdaFisher optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<AdaFisherParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                    state.fisher_diag_ema(torch::zeros_like(p));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();

                // Apply decoupled weight decay (AdamW-style)
                if (group_options.weight_decay() > 0.0) {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                auto& m = state.exp_avg();
                auto& F_diag = state.fisher_diag_ema();
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();

                // 1. Update the diagonal Fisher estimate (F_t)
                // F_t = beta2 * F_{t-1} + (1 - beta2) * g_t^2
                F_diag.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                // Bias correction for Fisher estimate
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);
                auto F_hat = F_diag / bias_correction2;

                // 2. Compute the preconditioned "natural" gradient
                // g_nat = g / (sqrt(F_hat) + eps)
                auto denom = F_hat.sqrt().add_(group_options.eps());
                auto natural_grad = grad / denom;

                // 3. Update momentum (m_t) on the natural gradient
                m.mul_(beta1).add_(natural_grad, 1.0 - beta1);

                // 4. Bias correction for momentum
                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                auto m_hat = m / bias_correction1;

                // 5. Final update
                p.data().add_(m_hat, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AdaFisher::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AdaFisher::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaFisher::make_param_state() { return std::make_unique<AdaFisherParamState>(); }
}