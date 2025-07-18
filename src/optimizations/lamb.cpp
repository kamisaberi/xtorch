#include <optimizations/lamb.h>
#include <stdexcept>
namespace xt::optim
{
    // --- LambOptions Methods ---
    void LambOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }

    void LambOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> LambOptions::clone() const {
        auto cloned = std::make_unique<LambOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- LambParamState Methods ---
    void LambParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }

    void LambParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> LambParamState::clone() const {
        auto cloned = std::make_unique<LambParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }


    // --- LAMB Implementation ---
    LAMB::LAMB(std::vector<torch::Tensor> params, LambOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<LambOptions>(options)) {}

    LAMB::LAMB(std::vector<torch::Tensor> params, double lr)
        : LAMB(std::move(params), LambOptions(lr)) {}

    torch::Tensor LAMB::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<LambOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("LAMB optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<LambParamState&>(*state_.at(p.unsafeGetTensorImpl()));

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

            // 1. Adam Core: Update biased moments
            m.mul_(beta1).add_(grad, 1.0 - beta1);
            v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

            // Bias correction
            double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
            double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;

            // This is the Adam update direction (before weight decay)
            auto adam_update = m_hat / (v_hat.sqrt() + group_options.eps());

            // 2. Add decoupled weight decay
            if (group_options.weight_decay() > 0.0) {
                adam_update.add_(p.detach(), group_options.weight_decay());
            }

            // 3. Calculate Trust Ratio
            // r_t = ||param||_2 / ||update||_2
            auto weight_norm = p.detach().norm(2).item<double>();
            auto update_norm = adam_update.norm(2).item<double>();

            double trust_ratio = 1.0;
            if (weight_norm > 0 && update_norm > 0) {
                trust_ratio = weight_norm / update_norm;
            }

            // 4. Calculate effective learning rate and apply final update
            // effective_lr = global_lr * trust_ratio
            // update = p - effective_lr * adam_update
            p.data().add_(adam_update, -group_options.lr() * trust_ratio);
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void LAMB::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void LAMB::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> LAMB::make_param_state() { return std::make_unique<LambParamState>(); }
}