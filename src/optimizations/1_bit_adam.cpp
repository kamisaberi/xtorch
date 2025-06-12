#include "include/optimizations/1_bit_adam.h"

namespace xt::optimizations
{
    // --- OneBitAdam Class Implementation ---

    OneBitAdam::OneBitAdam(std::vector<torch::Tensor> params, OneBitAdamOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<OneBitAdamOptions>(options))
    {
    }

    OneBitAdam::OneBitAdam(std::vector<torch::Tensor> params, double lr)
        : OneBitAdam(std::move(params), OneBitAdamOptions(lr))
    {
    }

    using LossClosure = std::function<torch::Tensor()>;
    torch::Tensor OneBitAdam::step(LossClosure closure)
    {
        torch::NoGradGuard no_grad;

        torch::Tensor loss = {};
        if (closure)
        {
            loss = closure();
        }

        for (auto& group : param_groups_)
        {
            auto& options = static_cast<OneBitAdamOptions&>(group.options());

            for (auto& p : group.params())
            {
                if (!p.grad().defined())
                {
                    continue;
                }

                auto grad = p.grad();
                if (grad.is_sparse())
                {
                    throw std::runtime_error("OneBitAdam does not support sparse gradients (yet).");
                }

                // --- THIS IS THE CORRECTED PART ---
                // Get the unique_ptr to the state
                auto& state_ptr = state_[p.unsafeGetTensorImpl()];

                // Initialize state if not already done.
                // If state_ptr is null, it means this param hasn't been seen before.
                // Optimizer::state_ automatically default-constructs a unique_ptr
                // if the key is not found, but it will be a unique_ptr to the base
                // OptimizerParamState if we don't handle its creation.
                // However, the `make_param_state` mechanism ensures that when
                // `state_[key]` is accessed and the state doesn't exist,
                // `add_param_group` (which is implicitly called when Optimizer is constructed)
                // or `load` will use `make_param_state` to create the correct derived type.
                // So, by the time we are here in `step`, `state_ptr` should already
                // point to a valid `OneBitAdamParamState` (or be null if this is the very first step
                // for a param that wasn't in the initial groups during construction, which is rare).
                // A more robust way (though usually not necessary if params are fixed at construction)
                // would be to check if state_ptr is null and create it.
                // However, the base class Optimizer usually ensures state is initialized for
                // all parameters passed to the constructor.

                if (!state_ptr)
                {
                    // Should ideally not happen if params are passed to constructor
                    // and optimizer has been initialized.
                    // This is a defensive check.
                    state_ptr = make_param_state(); // Ensure it's the correct type
                }

                // Now, safely cast the raw pointer obtained from unique_ptr::get()
                auto& state = static_cast<OneBitAdamParamState&>(*state_ptr.get());
                // --- END OF CORRECTION ---


                if (!state.step().defined())
                {
                    // This check might be redundant if state is always initialized by this point
                    state.step(torch::tensor(0.0, p.options()));
                    state.exp_avg(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                    state.exp_avg_sq(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                    state.error_feedback(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                    state.momentum_buffer(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                }

                auto& exp_avg = state.exp_avg();
                auto& exp_avg_sq = state.exp_avg_sq();
                auto& error_feedback = state.error_feedback();
                auto& momentum_buffer = state.momentum_buffer();

                state.step(state.step() + 1);
                double current_step_val = state.step().item<double>();

                if (options.weight_decay() != 0)
                {
                    grad = grad.add(p.detach(), options.weight_decay());
                }

                double beta1 = options.beta1();
                double beta2 = options.beta2();

                exp_avg.mul_(beta1).add_(grad, 1.0 - beta1);
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

                torch::Tensor m_t_hat;

                if (current_step_val <= options.warmup_steps())
                {
                    m_t_hat = exp_avg / bias_correction1;
                    momentum_buffer.copy_(m_t_hat);
                    error_feedback.zero_();
                }
                else
                {
                    auto m_t_compensated = (exp_avg / bias_correction1) + error_feedback;
                    momentum_buffer.copy_(m_t_compensated);

                    auto scale = momentum_buffer.abs().mean();
                    if (scale.item<double>() < 1e-10 && scale.item<double>() > -1e-10)
                    {
                        m_t_hat = torch::sign(momentum_buffer);
                    }
                    else
                    {
                        m_t_hat = torch::sign(momentum_buffer) * scale;
                    }
                    error_feedback.copy_(momentum_buffer - m_t_hat);
                }

                auto denom = (exp_avg_sq.sqrt() / std::sqrt(bias_correction2)).add_(options.eps());
                p.data().addcdiv_(m_t_hat, denom, -options.get_lr());
            }
        }
        return loss;
    }

    // ... (rest of the file: save, load, make_param_state)
    void OneBitAdam::save(torch::serialize::OutputArchive& archive) const
    {
        torch::optim::Optimizer::save(archive);
    }

    void OneBitAdam::load(torch::serialize::InputArchive& archive)
    {
        torch::optim::Optimizer::load(archive);
    }

    std::unique_ptr<torch::optim::OptimizerParamState> OneBitAdam::make_param_state()
    {
        return std::make_unique<OneBitAdamParamState>();
    }
}
