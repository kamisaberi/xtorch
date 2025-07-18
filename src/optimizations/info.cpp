#include <optimizations/info.h>
#include <stdexcept>
namespace xt::optim
{
    // --- InfoOptions Methods ---
    void InfoOptions::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("lr", this->lr());
        archive.write("beta1", beta1());
        archive.write("beta2", beta2());
        archive.write("fisher_beta", fisher_beta());
        archive.write("fisher_damping", fisher_damping());
        archive.write("eps", eps());
        archive.write("weight_decay", weight_decay());
    }

    void InfoOptions::deserialize(torch::serialize::InputArchive& archive) {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta1", ivalue)) { beta1_ = ivalue.toDouble(); }
        if (archive.try_read("beta2", ivalue)) { beta2_ = ivalue.toDouble(); }
        if (archive.try_read("fisher_beta", ivalue)) { fisher_beta_ = ivalue.toDouble(); }
        if (archive.try_read("fisher_damping", ivalue)) { fisher_damping_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> InfoOptions::clone() const {
        auto cloned = std::make_unique<InfoOptions>(this->lr());
        cloned->beta1(beta1()).beta2(beta2()).fisher_beta(fisher_beta())
              .fisher_damping(fisher_damping()).eps(eps()).weight_decay(weight_decay());
        return cloned;
    }

    // --- InfoParamState Methods ---
    void InfoParamState::serialize(torch::serialize::OutputArchive& archive) const {
        archive.write("step", step(), true);
        if(fisher_diag_ema().defined()) archive.write("fisher_diag_ema", fisher_diag_ema(), true);
        if(exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
        if(exp_avg_sq().defined()) archive.write("exp_avg_sq", exp_avg_sq(), true);
    }

    void InfoParamState::deserialize(torch::serialize::InputArchive& archive) {
        at::Tensor temp;
        if(archive.try_read("step", temp, true)) { step_ = temp; }
        if(archive.try_read("fisher_diag_ema", temp, true)) { fisher_diag_ema_ = temp; }
        if(archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
        if(archive.try_read("exp_avg_sq", temp, true)) { exp_avg_sq_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> InfoParamState::clone() const {
        auto cloned = std::make_unique<InfoParamState>();
        if(step().defined()) cloned->step(step().clone());
        if(fisher_diag_ema().defined()) cloned->fisher_diag_ema(fisher_diag_ema().clone());
        if(exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        if(exp_avg_sq().defined()) cloned->exp_avg_sq(exp_avg_sq().clone());
        return cloned;
    }

    // --- INFO Optimizer Implementation ---
    INFO::INFO(std::vector<torch::Tensor> params, InfoOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<InfoOptions>(options)) {}

    torch::Tensor INFO::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<InfoOptions&>(param_groups_[0].options());

        for (auto& p : param_groups_[0].params()) {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            auto& state = static_cast<InfoParamState&>(*state_.at(p.unsafeGetTensorImpl()));

            if (!state.step().defined()) {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.fisher_diag_ema(torch::zeros_like(p));
                state.exp_avg(torch::zeros_like(p));
                state.exp_avg_sq(torch::zeros_like(p));
            }
            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            // 1. Update Diagonal Fisher Information Matrix (FIM) Estimate
            // F_t = beta * F_{t-1} + (1 - beta) * g_t^2
            auto& fisher_diag = state.fisher_diag_ema();
            fisher_diag.mul_(group_options.fisher_beta()).addcmul_(grad, grad, 1.0 - group_options.fisher_beta());

            // 2. Compute the Natural Gradient
            // Natural Grad = F_t^{-1} * grad
            // For a diagonal FIM, this is just grad / (F_diag + damping)
            auto natural_grad = grad / (fisher_diag + group_options.fisher_damping());

            // 3. Apply Adam to the Natural Gradient
            auto& m = state.exp_avg();
            auto& v = state.exp_avg_sq();
            double bias_correction1 = 1.0 - std::pow(group_options.beta1(), current_step_val);
            double bias_correction2 = 1.0 - std::pow(group_options.beta2(), current_step_val);

            // Update momentum (m_t) on the natural gradient
            m.mul_(group_options.beta1()).add_(natural_grad, 1.0 - group_options.beta1());

            // Update adaptive learning rate (v_t) on the natural gradient
            v.mul_(group_options.beta2()).addcmul_(natural_grad, natural_grad, 1.0 - group_options.beta2());

            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;

            auto denom = v_hat.sqrt().add_(group_options.eps());

            // 4. Apply Decoupled Weight Decay & Final Update
            if (group_options.weight_decay() > 0.0) {
                p.data().add_(p.data(), -group_options.lr() * group_options.weight_decay());
            }

            // Final update uses the Adam-processed natural gradient
            p.data().addcdiv_(m_hat, denom, -group_options.lr());
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void INFO::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void INFO::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }
    std::unique_ptr<torch::optim::OptimizerParamState> INFO::make_param_state() { return std::make_unique<InfoParamState>(); }
}