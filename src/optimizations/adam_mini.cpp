#include <optimizations/adam_mini.h>
#include <stdexcept>

namespace xt::optim
{
    // --- AdamMiniOptions Methods ---
    void AdamMiniOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }
    void AdamMiniOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AdamMiniOptions::clone() const {
        auto cloned = std::make_unique<AdamMiniOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- AdamMiniParamState Methods ---
    void AdamMiniParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }
    void AdamMiniParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AdamMiniParamState::clone() const {
        auto cloned = std::make_unique<AdamMiniParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }


    // --- AdamMini Implementation ---
    AdamMini::AdamMini(std::vector<torch::Tensor> params, AdamMiniOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AdamMiniOptions>(options)) {}

    AdamMini::AdamMini(std::vector<torch::Tensor> params, double lr)
        : AdamMini(std::move(params), AdamMiniOptions(lr)) {}

    torch::Tensor AdamMini::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AdamMiniOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AdamMini optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<AdamMiniParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    // m_t is full precision
                    state.exp_avg(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                    // v_t is stored in half precision
                    state.exp_avg_sq(torch::zeros({p.sizes()}, p.options().dtype(torch::kFloat16)));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();

                // Apply decoupled weight decay
                if (group_options.weight_decay() > 0.0) {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                auto& m = state.exp_avg();
                auto& v_quantized = state.exp_avg_sq();
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();

                // 1. Update first moment (m_t) in full precision
                m.mul_(beta1).add_(grad, 1.0 - beta1);

                // 2. Dequantize v_t to float32 for update, then update and re-quantize
                //    This is the core memory-saving step.
                auto v = v_quantized.to(torch::kFloat32);
                v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);
                state.exp_avg_sq(v.to(torch::kFloat16)); // Store back in half precision

                // 3. Bias correction (using full-precision values)
                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

                auto m_hat = m / bias_correction1;
                auto v_hat = v / bias_correction2;

                // 4. Final update
                auto denom = v_hat.sqrt().add(group_options.eps());
                p.data().addcdiv_(m_hat, denom, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AdamMini::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AdamMini::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AdamMini::make_param_state() { return std::make_unique<AdamMiniParamState>(); }
}