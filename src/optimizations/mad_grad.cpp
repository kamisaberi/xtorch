#include <optimizations/mad_grad.h>
#include <stdexcept>

namespace xt::optim
{
    // --- MADGRADOptions Methods ---
    void MADGRADOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("momentum", momentum());
        archive.write("weight_decay", weight_decay());
        archive.write("eps", eps());
    }

    void MADGRADOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("momentum", ivalue)) { momentum_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> MADGRADOptions::clone() const {
        auto cloned = std::make_unique<MADGRADOptions>(this->lr());
        cloned->momentum(momentum()).weight_decay(weight_decay()).eps(eps());
        return cloned;
    }

    // --- MADGRADParamState Methods ---
    void MADGRADParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(grad_sum().defined()) archive.write("grad_sum", grad_sum(), true);
        if(grad_sum_sq().defined()) archive.write("grad_sum_sq", grad_sum_sq(), true);
        if(grad_prev().defined()) archive.write("grad_prev", grad_prev(), true);
    }

    void MADGRADParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("grad_sum", temp, true)) { grad_sum_ = temp; }
        if(archive.try_read("grad_sum_sq", temp, true)) { grad_sum_sq_ = temp; }
        if(archive.try_read("grad_prev", temp, true)) { grad_prev_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> MADGRADParamState::clone() const {
        auto cloned = std::make_unique<MADGRADParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(grad_sum().defined()) cloned->grad_sum(grad_sum().clone());
        if(grad_sum_sq().defined()) cloned->grad_sum_sq(grad_sum_sq().clone());
        if(grad_prev().defined()) cloned->grad_prev(grad_prev().clone());
        return cloned;
    }


    // --- MADGRAD Implementation ---
    MADGRAD::MADGRAD(std::vector<torch::Tensor> params, MADGRADOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<MADGRADOptions>(options)) {}

    MADGRAD::MADGRAD(std::vector<torch::Tensor> params, double lr)
        : MADGRAD(std::move(params), MADGRADOptions(lr)) {}


    torch::Tensor MADGRAD::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<MADGRADOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse()) {
                throw std::runtime_error("MADGRAD optimizer does not support sparse gradients.");
            }

            auto& state = static_cast<MADGRADParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.grad_sum(torch::zeros_like(p));
                state.grad_sum_sq(torch::zeros_like(p));
                state.grad_prev(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);

            // Apply decoupled weight decay
            if (group_options.weight_decay() > 0.0) {
                p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
            }

            auto& s = state.grad_sum();
            auto& v = state.grad_sum_sq();
            auto& g_prev = state.grad_prev();
            double momentum = group_options.momentum();

            // 1. Update first moment (s_k)
            // s_k = lambda_k * s_{k-1} + g_k
            // We use a fixed lambda (momentum) for simplicity
            s.mul_(momentum).add_(grad);

            // 2. Update second moment (v_k)
            // v_k = lambda_k * v_{k-1} + (g_k - lambda_k * g_{k-1}) * g_k
            // This term `(g_k - lambda_k * g_{k-1})` captures the "volatility"
            auto grad_volatility = grad - (g_prev * momentum);
            v.mul_(momentum).addcmul_(grad, grad_volatility, 1.0);

            // Store current gradient for next step's volatility calculation
            g_prev.copy_(grad);

            // 3. Compute the update
            // Denominator: (v_k)^(1/3) + eps
            // We add eps before the root for numerical stability if v is negative (though it shouldn't be)
            // or zero.
            auto v_cbrt = (v.abs() + group_options.eps()).pow(1.0/3.0);

            // Final update: p_k = p_{k-1} - lr * s_k / (v_k^(1/3) + eps)
            // Note: some MADGRAD implementations might add eps after the root.
            // Adding before is safer. The paper suggests adding to v_k.
            p.data().addcdiv_(s, v_cbrt, -group_options.lr());
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void MADGRAD::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void MADGRAD::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> MADGRAD::make_param_state() { return std::make_unique<MADGRADParamState>(); }
}