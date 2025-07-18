#include <optimizations/ada_shift.h>
#include <stdexcept>

namespace xt::optim
{
    // --- AdaShiftOptions Methods ---
    void AdaShiftOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
        archive.write("lookback_d", lookback_d());
    }
    void AdaShiftOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("lookback_d", ivalue)) { lookback_d_ = ivalue.toInt(); }
    }
    std::unique_ptr<torch::optim::OptimizerOptions> AdaShiftOptions::clone() const {
        auto cloned = std::make_unique<AdaShiftOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).eps(eps())
              .weight_decay(weight_decay()).lookback_d(lookback_d());
        return cloned;
    }

    // --- AdaShiftParamState Methods ---
    void AdaShiftParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
        archive.write("history_size", static_cast<int64_t>(grad_sq_history.size()));
        for (size_t i = 0; i < grad_sq_history.size(); ++i) {
            archive.write("grad_sq_hist_" + std::to_string(i), grad_sq_history[i], true);
        }
    }
    void AdaShiftParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
        c10::IValue ivalue;
        if (archive.try_read("history_size", ivalue)) {
            int64_t history_size = ivalue.toInt();
            grad_sq_history.resize(history_size);
            for (int64_t i = 0; i < history_size; ++i) {
                archive.read("grad_sq_hist_" + std::to_string(i), grad_sq_history[i], true);
            }
        }
    }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaShiftParamState::clone() const {
        auto cloned = std::make_unique<AdaShiftParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        for(const auto& g_sq : grad_sq_history) cloned->grad_sq_history.push_back(g_sq.clone());
        return cloned;
    }


    // --- AdaShift Implementation ---
    AdaShift::AdaShift(std::vector<torch::Tensor> params, AdaShiftOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<AdaShiftOptions>(options)) {}

    AdaShift::AdaShift(std::vector<torch::Tensor> params, double lr)
        : AdaShift(std::move(params), AdaShiftOptions(lr)) {}

    torch::Tensor AdaShift::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<AdaShiftOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("AdaShift optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<AdaShiftParamState&>(*state_.at(p.unsafeGetTensorImpl()));

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
                auto& grad_sq_history = state.grad_sq_history;
                double beta1 = group_options.beta1();
                double beta2 = group_options.beta2();

                // 1. Update first moment (m_t) with the CURRENT gradient
                m.mul_(beta1).add_(grad, 1.0 - beta1);

                // 2. Store current squared gradient in history
                grad_sq_history.push_back(grad.square());

                // 3. Update second moment (v_t) with a DELAYED gradient
                if (grad_sq_history.size() > group_options.lookback_d()) {
                    auto delayed_grad_sq = grad_sq_history.front();
                    grad_sq_history.pop_front();

                    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_{t-d}^2
                    v.mul_(beta2).add_(delayed_grad_sq, 1.0 - beta2);
                }

                // 4. Bias correction and update
                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                auto m_hat = m / bias_correction1;

                // Only apply adaptive part if v has been updated at least once
                if (current_step_val > group_options.lookback_d()) {
                    // We use a different step counter for v's bias correction
                    double v_step = current_step_val - group_options.lookback_d();
                    double bias_correction2 = 1.0 - std::pow(beta2, v_step);

                    auto v_hat = v / bias_correction2;
                    auto denom = v_hat.sqrt().add(group_options.eps());
                    p.data().addcdiv_(m_hat, denom, -group_options.lr());
                } else {
                    // Before we have enough history, behave like Adam with momentum only (no v_t)
                    p.data().add_(m_hat, -group_options.lr());
                }
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void AdaShift::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void AdaShift::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> AdaShift::make_param_state() { return std::make_unique<AdaShiftParamState>(); }
}