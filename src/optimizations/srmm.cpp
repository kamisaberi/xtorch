#include <optimizations/srmm.h>
#include <stdexcept>

namespace xt::optim
{
    // --- SRMMOptions Methods ---
    void SRMMOptions::serialize(torch::serialize::OutputArchive& archive) const
    {
        archive.write("lr", this->lr());
        archive.write("beta", beta());
        archive.write("weight_decay", weight_decay());
        archive.write("eps", eps());
    }

    void SRMMOptions::deserialize(torch::serialize::InputArchive& archive)
    {
        c10::IValue ivalue;
        if (archive.try_read("lr", ivalue)) { this->lr(ivalue.toDouble()); }
        if (archive.try_read("beta", ivalue)) { beta_ = ivalue.toDouble(); }
        if (archive.try_read("weight_decay", ivalue)) { weight_decay_ = ivalue.toDouble(); }
        if (archive.try_read("eps", ivalue)) { eps_ = ivalue.toDouble(); }
    }

    std::unique_ptr<torch::optim::OptimizerOptions> SRMMOptions::clone() const
    {
        auto cloned = std::make_unique<SRMMOptions>(this->lr());
        cloned->beta(beta()).weight_decay(weight_decay()).eps(eps());
        return cloned;
    }

    // --- SRMMParamState Methods ---
    void SRMMParamState::serialize(torch::serialize::OutputArchive& archive) const
    {
        archive.write("step", step(), true);
        if (exp_avg().defined()) archive.write("exp_avg", exp_avg(), true);
    }

    void SRMMParamState::deserialize(torch::serialize::InputArchive& archive)
    {
        at::Tensor temp;
        if (archive.try_read("step", temp, true)) { step_ = temp; }
        if (archive.try_read("exp_avg", temp, true)) { exp_avg_ = temp; }
    }

    std::unique_ptr<torch::optim::OptimizerParamState> SRMMParamState::clone() const
    {
        auto cloned = std::make_unique<SRMMParamState>();
        if (step().defined()) cloned->step(step().clone());
        if (exp_avg().defined()) cloned->exp_avg(exp_avg().clone());
        return cloned;
    }

    // --- SRMM Implementation ---
    SRMM::SRMM(std::vector<torch::Tensor> params, SRMMOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<SRMMOptions>(options))
    {
    }

    SRMM::SRMM(std::vector<torch::Tensor> params, double lr)
        : SRMM(std::move(params), SRMMOptions(lr))
    {
    }

    torch::Tensor SRMM::step(LossClosure closure)
    {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        auto& group_options = static_cast<SRMMOptions&>(param_groups_[0].options());

        for (auto& group : param_groups_)
        {
            for (auto& p : group.params())
            {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse())
                {
                    throw std::runtime_error("SRMM optimizer does not support sparse gradients.");
                }

                auto& state = static_cast<SRMMParamState&>(*state_.at(p.unsafeGetTensorImpl()));

                if (!state.step().defined())
                {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p));
                }
                state.step(state.step() + 1.0);
                double current_step_val = state.step().item<double>();

                // Apply decoupled weight decay
                if (group_options.weight_decay() > 0.0)
                {
                    p.data().mul_(1.0 - group_options.lr() * group_options.weight_decay());
                }

                // 1. Update momentum (m_t)
                auto& m = state.exp_avg();
                m.mul_(group_options.beta()).add_(grad, 1.0 - group_options.beta());

                // 2. Apply bias correction (important for early steps)
                double bias_correction = 1.0 - std::pow(group_options.beta(), current_step_val);
                auto m_hat = m / bias_correction;

                // 3. Compute the Square-Root Momentum update
                // update = m_hat / sqrt(||m_hat||_2)
                auto m_hat_norm = m_hat.norm(2);

                torch::Tensor update;
                if (m_hat_norm.item<double>() > group_options.eps())
                {
                    // Denominator is sqrt of the L2 norm
                    auto denom = m_hat_norm.sqrt().add(group_options.eps());
                    update = m_hat / denom;
                }
                else
                {
                    // If norm is zero, update is zero
                    update = torch::zeros_like(p);
                }

                // 4. Apply the final update
                p.data().add_(update, -group_options.lr());
            }
        }
        return loss;
    }

    // --- Boilerplate Methods ---
    void SRMM::save(torch::serialize::OutputArchive& archive) const { torch::optim::Optimizer::save(archive); }
    void SRMM::load(torch::serialize::InputArchive& archive) { torch::optim::Optimizer::load(archive); }

    std::unique_ptr<torch::optim::OptimizerParamState> SRMM::make_param_state()
    {
        return std::make_unique<SRMMParamState>();
    }
}