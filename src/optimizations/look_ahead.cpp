#include <optimizations/look_ahead.h>
#include <stdexcept>

namespace xt::optim
{
    // --- LookaheadOptions Methods ---
    void LookaheadOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("alpha", alpha());
        archive.write("k", k());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }

    void LookaheadOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("alpha", ivalue)) { alpha_ = ivalue.toDouble(); }
        if (archive.try_read("k", ivalue)) { k_ = ivalue.toInt(); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> LookaheadOptions::clone() const {
        auto cloned = std::make_unique<LookaheadOptions>(this->lr());
        cloned->alpha(alpha()).k(k()).beta1(beta1()).beta2(beta2())
              .eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- LookaheadParamState Methods ---
    void LookaheadParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(slow_param().defined()) archive.write("slow_param", slow_param(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }

    void LookaheadParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("slow_param", temp, true)) { slow_param_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> LookaheadParamState::clone() const {
        auto cloned = std::make_unique<LookaheadParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(slow_param().defined()) cloned->slow_param(slow_param().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }

    // --- Lookahead Implementation ---
    Lookahead::Lookahead(std::vector<torch::Tensor> params, LookaheadOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<LookaheadOptions>(options)) {

        // Initialize the "slow" parameters to be a copy of the initial model parameters
        for (auto& group : param_groups_) {
            for (auto& p : group.params()) {
                // make_param_state() is called implicitly by the base constructor for each param
                auto& state = static_cast<LookaheadParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                state.slow_param(p.detach().clone());
            }
        }
    }

    torch::Tensor Lookahead::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        // --- THIS IS THE CORRECTED LINE ---
        auto& group_options = static_cast<LookaheadOptions&>(param_groups_[0].options());

        // This step count is the *inner* optimizer's step count
        long global_step = 0;

        for (auto& group : param_groups_) { // Iterate through param groups
            for (auto& p : group.params()) { // Iterate through params in the group
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("Lookahead optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<LookaheadParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                    state.exp_avg_sq(torch::zeros_like(p));
                    // 'slow_param' should have been initialized in the constructor
                    if (!state.slow_param().defined()) {
                        state.slow_param(p.detach().clone());
                    }
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();
                if (global_step == 0) global_step = static_cast<long>(current_step_val);

                // --- Step 1: Perform an inner Adam optimizer step ---
                auto& m = state.exp_avg();
                auto& v = state.exp_avg_sq();
                double bias_correction1 = 1.0 - std::pow(group_options.beta1(), current_step_val);
                double bias_correction2 = 1.0 - std::pow(group_options.beta2(), current_step_val);

                torch::Tensor grad_with_wd = grad;
                if (group_options.weight_decay() > 0.0) {
                    grad_with_wd = grad.add(p.detach(), group_options.weight_decay());
                }

                m.mul_(group_options.beta1()).add_(grad_with_wd, 1.0 - group_options.beta1());
                v.mul_(group_options.beta2()).addcmul_(grad_with_wd, grad_with_wd, 1.0 - group_options.beta2());

                auto m_hat = m / bias_correction1;
                auto v_hat = v / bias_correction2;
                auto denom = v_hat.sqrt().add_(group_options.eps());

                p.data().addcdiv_(m_hat, denom, -group_options.lr());
            }
        }

        // --- Step 2: Check for Lookahead synchronization ---
        if (global_step > 0 && global_step % group_options.k() == 0) {
            for (auto& group : param_groups_) {
                for (auto& p : group.params()) {
                    auto& state = static_cast<LookaheadParamState&>(*state_.at(p.unsafeGetTensorImpl()));
                    auto& slow_p = state.slow_param();

                    // Update slow weights: slow_p += alpha * (fast_p - slow_p)
                    slow_p.add_(p.detach() - slow_p, group_options.alpha());

                    // Sync fast weights back to the new slow weights
                    p.data().copy_(slow_p);
                }
            }
        }

        return loss;
    }

    // --- Boilerplate Methods ---
    void Lookahead::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void Lookahead::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> Lookahead::make_param_state() { return std::make_unique<LookaheadParamState>(); }
}