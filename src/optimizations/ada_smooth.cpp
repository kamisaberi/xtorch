#include <optimizations/ada_smooth.h>
#include <stdexcept>

namespace xt::optim
{
    // --- AdaSmoothOptions Methods ---
    void AdaSmoothOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("beta3", beta3());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }
    void AdaSmoothOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("beta3", ivalue)) { beta3_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AdaSmoothOptions::clone() const {
        auto cloned = std::make_unique<AdaSmoothOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).beta3(beta3()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- AdaSmoothParamState Methods ---
    void AdaSmoothParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
        if(smooth_update().defined()) archive.write("smooth_update", smooth_update(), true);
    }
    void AdaSmoothParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
        if(archive.try_read("smooth_update", temp, true)) { smooth_update_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaSmoothParamState::clone() const {
        auto cloned = std::make_unique<AdaSmoothParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        if(smooth_update().defined()) cloned->smooth_update(smooth_update().clone());
        return cloned;
    }

    // --- AdaSmooth Implementation ---
    AdaSmooth::AdaSmooth(std::vector<torch::Tensor> params, AdaSmoothOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AdaSmoothOptions>(options)) {}

    AdaSmooth::AdaSmooth(std::vector<torch::Tensor> params, double lr)
        : AdaSmooth(std::move(params), AdaSmoothOptions(lr)) {}

    torch::Tensor AdaSmooth::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AdaSmoothOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AdaSmooth optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<AdaSmoothParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                    state.exp_avg_sq(torch::zeros_like(p));
                    state.smooth_update(torch::zeros_like(p));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();

                // Apply decoupled weight decay
                if (group_options.weight_decay() > 0.0) {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                auto& m = state.exp_avg();
                auto& v = state.exp_avg_sq();
                auto& s = state.smooth_update();
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();
                double beta3 = group_options.beta3();

                // 1. Standard Adam moment updates
                m.mul_(beta1).add_(grad, 1.0 - beta1);
                v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

                auto m_hat = m / bias_correction1;
                auto v_hat = v / bias_correction2;

                // 2. Compute the instantaneous Adam step
                auto denom = v_hat.sqrt().add(group_options.eps());
                auto adam_step = (m_hat / denom) * group_options.lr();

                // 3. Update the smoothed update step (s_t)
                // s_t = beta3 * s_{t-1} + (1 - beta3) * adam_step_t
                s.mul_(beta3).add_(adam_step, 1.0 - beta3);

                // 4. Apply the final, smoothed update
                p.data().sub_(s);
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AdaSmooth::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AdaSmooth::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaSmooth::make_param_state() { return std::make_unique<AdaSmoothParamState>(); }
}