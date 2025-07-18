#include <optimizations/qh_adam.h>
#include <stdexcept>

namespace xt::optim
{
    // --- QHAdamOptions Methods ---
    void QHAdamOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("nu1", nu1());
        archive.write("nu2", nu2());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("weight_decay", weight_decay());
        archive.write("eps", eps());
    }
    void QHAdamOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("nu1", ivalue)) { nu1_ = ivalue.toDouble(); }
        if (archive.try_read("nu2", ivalue)) { nu2_ = ivalue.toDouble(); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> QHAdamOptions::clone() const {
        auto cloned = std::make_unique<QHAdamOptions>(this->lr());
        cloned->nu1(nu1()).nu2(nu2()).beta1(beta1()).beta2(beta2())
              .weight_decay(weight_decay()).eps(eps());
        return cloned;
    }

    // --- QHAdamParamState Methods ---
    void QHAdamParamState::serialize(torch::serialize::OutputArchive& archive) const {
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }
    void QHAdamParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> QHAdamParamState::clone() const {
        auto cloned = std::make_unique<QHAdamParamState>();
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }

    // --- QHAdam Implementation ---
    QHAdam::QHAdam(std::vector<torch::Tensor> params, QHAdamOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<QHAdamOptions>(options)) {}

    QHAdam::QHAdam(std::vector<torch::Tensor> params, double lr)
        : QHAdam(std::move(params), QHAdamOptions(lr)) {}

    torch::Tensor QHAdam::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<QHAdamOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("QHAdam optimizer does not support sparse gradients.");
                }

                // Apply decoupled weight decay
                if (group_options.weight_decay() > 0.0) {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                auto& state = static_cast<QHAdamParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                if (!state.exp_avg().defined()) {
                    state.exp_avg(torch::zeros_like(p));
                    state.exp_avg_sq(torch::zeros_like(p));
                }

                auto& m = state.exp_avg();
                auto& v = state.exp_avg_sq();
                double nu1 = group_options.nu1();
                double nu2 = group_options.nu2();
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();

                // 1. Update first moment (m_t)
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                m.mul_(beta1).add_(grad, 1.0 - beta1);

                // 2. Update second moment (v_t)
                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                v.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                // 3. Construct the quasi-hyperbolic update
                // Numerator: (1 - nu1) * g_t + nu1 * m_t
                auto numerator = grad * (1.0 - nu1) + m * nu1;

                // Denominator: sqrt( (1 - nu2) * g_t^2 + nu2 * v_t )
                // We use the common RMSProp-like version where nu2=1.0, so this simplifies.
                torch::Tensor denom;
                if (nu2 == 1.0) {
                    denom = v.sqrt().add_(group_options.eps());
                } else {
                    auto v_component = v * nu2;
                    auto g_sq_component = grad.square() * (1.0 - nu2);
                    denom = (v_component + g_sq_component).sqrt().add_(group_options.eps());
                }

                // 4. Final update
                p.data().addcdiv_(numerator, denom, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void QHAdam::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void QHAdam::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> QHAdam::make_param_state() { return std::make_unique<QHAdamParamState>(); }
}