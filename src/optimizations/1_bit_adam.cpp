#include "include/optimizations/1_bit_adam.h" // Your header file
#include <stdexcept>       // For std::runtime_error

// --- OneBitAdam Class Implementation ---

OneBitAdam::OneBitAdam(std::vector<torch::Tensor> params, OneBitAdamOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<OneBitAdamOptions>(options))
{
}

OneBitAdam::OneBitAdam(std::vector<torch::Tensor> params, double lr_val)
    : OneBitAdam(std::move(params), OneBitAdamOptions(lr_val))
{
}

// Note: The LossClosure type alias is defined in your header now, which is good.
// The override keyword in the header for step, save, load, make_param_state is also correct.
torch::Tensor OneBitAdam::step(LossClosure closure)
{
    torch::NoGradGuard no_grad; // Optimizer steps should not track gradients

    torch::Tensor loss = {};
    if (closure)
    {
        loss = closure();
    }

    for (auto& group : param_groups_)
    {
        // The options for this group, cast to our specific type.
        // This is safe because we passed OneBitAdamOptions to the Optimizer constructor.
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
                // You could choose to support sparse gradients later if needed,
                // but it requires different handling for exp_avg and exp_avg_sq.
                throw std::runtime_error("OneBitAdam does not support sparse gradients at this time.");
            }

            // Get the parameter-specific state.
            // state_ is a map from TensorImpl* to unique_ptr<OptimizerParamState>.
            // The .at() method will throw std::out_of_range if the param is not found,
            // which shouldn't happen if parameters were correctly added to the optimizer.
            auto& state_ptr = state_.at(p.unsafeGetTensorImpl());
            // Cast the base OptimizerParamState pointer to our derived type.
            // This is safe because make_param_state() ensures the correct type is created.
            auto& state = static_cast<OneBitAdamParamState&>(*state_ptr.get());


            // Initialize state tensors if this is the first step for this parameter.
            if (!state.step().defined())
            {
                // Initialize step as a CPU scalar tensor of type double.
                // .item<double>() requires CPU tensor.
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                // Other state tensors are initialized based on the parameter's properties.
                state.exp_avg(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                state.exp_avg_sq(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                state.error_feedback(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                state.momentum_buffer(torch::zeros_like(p, torch::MemoryFormat::Preserve));
            }

            // Retrieve references to state tensors for convenience.
            auto& exp_avg = state.exp_avg();
            auto& exp_avg_sq = state.exp_avg_sq();
            auto& error_feedback = state.error_feedback();
            auto& momentum_buffer = state.momentum_buffer();

            // Increment step count. Ensure it remains a CPU scalar.
            torch::Tensor current_step_cpu = state.step(); // Should already be CPU scalar
            // TORCH_CHECK(current_step_cpu.is_cpu() && current_step_cpu.isscalar(), "Step tensor is not a CPU scalar!"); // Optional debug check

            current_step_cpu += 1.0;
            state.step(current_step_cpu);
            double current_step_val = current_step_cpu.item<double>(); // This is step_t for the current update


            // Apply weight decay (L2 regularization) if specified.
            // Weight decay is applied directly to the gradient.
            if (options.weight_decay() != 0.0)
            {
                grad = grad.add(p.detach(), options.weight_decay());
            }

            double beta1 = options.beta1();
            double beta2 = options.beta2();

            // Adam: Update biased first moment estimate (m_t)
            exp_avg.mul_(beta1).add_(grad, 1.0 - beta1);
            // Adam: Update biased second raw moment estimate (v_t)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

            // Adam: Compute bias-corrected first moment estimate (m_hat_t)
            // Bias correction term for the first moment.
            double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
            // Adam: Compute bias-corrected second raw moment estimate (v_hat_t)
            // Bias correction term for the second moment.
            double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

            torch::Tensor m_t_hat_for_update; // This will be the (possibly compressed) momentum used for the update

            if (current_step_val <= options.warmup_steps())
            {
                // During warmup: use full precision momentum
                m_t_hat_for_update = exp_avg / bias_correction1;
                momentum_buffer.copy_(m_t_hat_for_update); // Store full precision m_t
                error_feedback.zero_(); // Reset error feedback during warmup
            }
            else
            {
                // After warmup: apply 1-bit compression with error feedback
                // 1. Compensate m_t with previous error: m_t_comp = (m_t / bias_corr1) + e_{t-1}
                auto m_t_compensated = (exp_avg / bias_correction1) + error_feedback;
                momentum_buffer.copy_(m_t_compensated); // Store for current error calculation

                // 2. Compress: m_hat_t = sign(m_t_comp) * scale_factor
                //    Scale factor is often mean of absolute values of m_t_comp.
                auto scale = momentum_buffer.abs().mean();

                // Prevent scale from being too small (prevents NaN/Inf during division or if m_t_comp is all zeros).
                // A very small epsilon could be added to scale, or use torch::clamp_min.
                // Or, if scale is effectively zero, the compressed momentum is effectively zero or just sign.
                if (scale.item<double>() < 1e-10 && scale.item<double>() > -1e-10)
                {
                    m_t_hat_for_update = torch::sign(momentum_buffer);
                    // Or torch::zeros_like(momentum_buffer) if scale is truly zero
                }
                else
                {
                    m_t_hat_for_update = torch::sign(momentum_buffer) * scale;
                }

                // 3. Calculate new error feedback for next step: e_t = m_t_comp - m_hat_t
                error_feedback.copy_(momentum_buffer - m_t_hat_for_update);
            }

            // Adam: Denominator for the update rule: sqrt(v_hat_t) / sqrt(bias_corr2) + eps
            auto denom = (exp_avg_sq.sqrt() / std::sqrt(bias_correction2)).add_(options.eps());

            // Parameter update: p_t = p_{t-1} - lr * m_hat_t_for_update / denom
            // options.lr() correctly gets the learning rate from the OptimizerOptions base.
            p.data().addcdiv_(m_t_hat_for_update, denom, -options.lr());
        }
    }
    return loss;
}

// Save optimizer state
void OneBitAdam::save(torch::serialize::OutputArchive& archive) const
{
    // The base Optimizer::save method handles serializing param_groups (including their options)
    // and the state_ map (by calling serialize on each OptimizerParamState).
    torch::optim::Optimizer::save(archive);
}

// Load optimizer state
void OneBitAdam::load(torch::serialize::InputArchive& archive)
{
    // The base Optimizer::load method handles deserializing param_groups and options.
    // For the state_ map, it uses make_param_state() to create state objects of the correct
    // derived type before calling deserialize on them.
    torch::optim::Optimizer::load(archive);
}

// Factory method for creating parameter-specific states.
// This is crucial for the load() method to instantiate the correct state type.
std::unique_ptr<torch::optim::OptimizerParamState> OneBitAdam::make_param_state()
{
    return std::make_unique<OneBitAdamParamState>();
}
