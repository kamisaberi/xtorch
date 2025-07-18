#include <optimizations/1_bit_lamb.h>
#include <stdexcept> // For std::runtime_error

namespace xt::optim
{
    // --- OneBitLamb Class Implementation ---

    OneBitLamb::OneBitLamb(std::vector<torch::Tensor> params, OneBitLambOptions options)
        : torch::optim::Optimizer(std::move(params), std::make_unique<OneBitLambOptions>(options)) {}

    OneBitLamb::OneBitLamb(std::vector<torch::Tensor> params, double lr_val)
        : OneBitLamb(std::move(params), OneBitLambOptions(lr_val)) {}

    torch::Tensor OneBitLamb::step(LossClosure closure) {
        torch::NoGradGuard no_grad;
        torch::Tensor loss = {};
        if (closure) { loss = closure(); }

        for (auto& group : param_groups_) {
            auto& options = static_cast<OneBitLambOptions&>(group.options());

            for (auto& p : group.params()) {
                if (!p.grad().defined()) { continue; }

                auto grad = p.grad();
                if (grad.is_sparse()) {
                    throw std::runtime_error("OneBitLamb does not support sparse gradients.");
                }

                auto& state_ptr = state_.at(p.unsafeGetTensorImpl());
                auto& state = static_cast<OneBitLambParamState&>(*state_ptr.get());

                if (!state.step().defined()) {
                    state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                    state.exp_avg(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                    state.exp_avg_sq(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                    state.error_feedback(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                    state.momentum_buffer(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                }

                auto& exp_avg = state.exp_avg();
                auto& exp_avg_sq = state.exp_avg_sq();
                auto& error_feedback = state.error_feedback();
                auto& momentum_buffer = state.momentum_buffer();

                torch::Tensor current_step_cpu = state.step();
                current_step_cpu += 1.0;
                state.step(current_step_cpu);
                double current_step_val = current_step_cpu.item<double>();

                double beta1 = options.beta1();
                double beta2 = options.beta2();

                // Adam part: Update biased moments
                exp_avg.mul_(beta1).add_(grad, 1.0 - beta1);
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1.0 - beta2);

                double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
                double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

                // Corrected m_t (exp_avg / bias_correction1)
                torch::Tensor m_t_corrected = exp_avg / bias_correction1;

                // 1-Bit Compression Logic (for m_t_corrected)
                torch::Tensor m_t_for_adam_update; // This is m_hat_t in Adam/LAMB context before LAMB scaling

                if (current_step_val <= options.warmup_steps()) {
                    m_t_for_adam_update = m_t_corrected;
                    momentum_buffer.copy_(m_t_for_adam_update);
                    error_feedback.zero_();
                } else {
                    auto m_t_compensated = m_t_corrected + error_feedback;
                    momentum_buffer.copy_(m_t_compensated);

                    auto scale = momentum_buffer.abs().mean();
                    if (scale.item<double>() < 1e-10 && scale.item<double>() > -1e-10) {
                        m_t_for_adam_update = torch::sign(momentum_buffer);
                    } else {
                        m_t_for_adam_update = torch::sign(momentum_buffer) * scale;
                    }
                    error_feedback.copy_(momentum_buffer - m_t_for_adam_update);
                }

                // Adam/LAMB: Denominator for the update (v_hat_t)
                torch::Tensor v_hat_t = exp_avg_sq / bias_correction2;
                torch::Tensor adam_update_direction = m_t_for_adam_update / (v_hat_t.sqrt() + options.eps());

                // LAMB specific: Add weight decay to the update direction before calculating trust ratio
                // update = m_hat / (sqrt(v_hat) + eps) + wd * p
                torch::Tensor lamb_update_direction = adam_update_direction;
                if (options.weight_decay() != 0.0) {
                    lamb_update_direction = lamb_update_direction.add(p.detach(), options.weight_decay());
                }

                // LAMB specific: Calculate trust ratio
                // r_t = ||p||_2 / ||update||_2
                // Add a small epsilon to norms to prevent division by zero if either norm is zero.
                // A very small weight norm or update norm can lead to NaN/Inf.
                // Using a minimum norm value can help stabilize.
                double param_norm_val = p.detach().norm(2).item<double>();
                double update_norm_val = lamb_update_direction.norm(2).item<double>();

                double trust_ratio = 1.0; // Default if norms are zero or problematic

                // Ensure norms are not too small to cause issues
                const double min_norm = 1e-6; // A small constant to avoid division by zero
                if (param_norm_val > min_norm && update_norm_val > min_norm) {
                    trust_ratio = param_norm_val / update_norm_val;
                } else if (param_norm_val > min_norm && update_norm_val <= min_norm) { // Update is zero, param is not
                    trust_ratio = options.trust_clip_threshold() > 0 ? options.trust_clip_threshold() : 100.0; // Large ratio
                } else { // Param is zero or both are zero
                    trust_ratio = 1.0; // No update or no change
                }


                // Clip trust ratio if a threshold is set
                if (options.trust_clip_threshold() > 0.0) {
                    trust_ratio = std::min(trust_ratio, options.trust_clip_threshold());
                }

                // Effective learning rate for this parameter
                double effective_lr = options.lr * trust_ratio;

                // Parameter update: p_t = p_{t-1} - effective_lr * lamb_update_direction
                p.data().add_(lamb_update_direction, -effective_lr);
            }
        }
        return loss;
    }

    void OneBitLamb::save(torch::serialize::OutputArchive& archive) const {
        torch::optim::Optimizer::save(archive);
    }

    void OneBitLamb::load(torch::serialize::InputArchive& archive) {
        torch::optim::Optimizer::load(archive);
    }

    std::unique_ptr<torch::optim::OptimizerParamState> OneBitLamb::make_param_state() {
        return std::make_unique<OneBitLambParamState>();
    }
}