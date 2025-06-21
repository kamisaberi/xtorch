#include "include/optimizations/ams_grad.h"
#include <stdexcept>
namespace xt::optim
{
    // --- AMSGradOptions Methods ---
    void AMSGradOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }
    void AMSGradOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AMSGradOptions::clone() const {
        auto cloned = std::make_unique<AMSGradOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- AMSGradParamState Methods ---
    void AMSGradParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
        if(max_exp_avg_sq().defined()) archive.write("max_exp_avg_sq", max_exp_avg_sq(), true);
    }
    void AMSGradParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
        if(archive.try_read("max_exp_avg_sq", temp, true)) { max_exp_avg_sq_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AMSGradParamState::clone() const {
        auto cloned = std::make_unique<AMSGradParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        if(max_exp_avg_sq().defined()) cloned->max_exp_avg_sq(max_exp_avg_sq().clone());
        return cloned;
    }


    // --- AMSGrad Implementation ---
    AMSGrad::AMSGrad(std::vector<torch::Tensor> params, AMSGradOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AMSGradOptions>(options)) {}

    AMSGrad::AMSGrad(std::vector<torch::Tensor> params, double lr)
        : AMSGrad(std::move(params), AMSGradOptions(lr)) {}

    torch::Tensor AMSGrad::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AMSGradOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AMSGrad optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<AMSGradParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                    state.exp_avg_sq(torch::zeros_like(p));
                    state.max_exp_avg_sq(torch::zeros_like(p)); // Initialize max_v
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();

                // Apply classic L2 regularization (weight decay)
                if (group_options.weight_decay() > 0.0) {
                    grad = grad.add(p.detach(), group_options.weight_decay());
                }

                auto& m = state.exp_avg();
                auto& v = state.exp_avg_sq();
                auto& max_v = state.max_exp_avg_sq();
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();

                // 1. Standard moment updates
                m.mul_(beta1).add_(grad, 1.0 - beta1);
                v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                // 2. AMSGrad modification: update max_v
                // max_v_t = max(max_v_{t-1}, v_t)
                torch::max_out(max_v, max_v, v);

                // 3. Bias correction
                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                // Note: The denominator uses max_v which is NOT bias corrected.
                // The bias correction is applied to m_t only.
                // The paper shows bias correction can be applied to max_v, but many
                // implementations (like PyTorch's) apply it to v before the max.
                // Let's follow the PyTorch model for consistency.
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);
                auto v_hat = v / bias_correction2;
                torch::max_out(max_v, max_v, v_hat); // Let's use the bias-corrected v for max comparison

                auto m_hat = m / bias_correction1;

                // 4. Final update using the max_v for the denominator
                auto denom = max_v.sqrt().add(group_options.eps());

                p.data().addcdiv_(m_hat, denom, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AMSGrad::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AMSGrad::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AMSGrad::make_param_state() { return std::make_unique<AMSGradParamState>(); }
}