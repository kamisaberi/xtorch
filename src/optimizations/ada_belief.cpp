#include "include/optimizations/ada_belief.h"
#include <stdexcept> // For std::runtime_error

// --- AdaBelief Class Implementation ---

AdaBelief::AdaBelief(std::vector<torch::Tensor> params, AdaBeliefOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<AdaBeliefOptions>(options))
{
}

AdaBelief::AdaBelief(std::vector<torch::Tensor> params, double lr_val)
    : AdaBelief(std::move(params), AdaBeliefOptions(lr_val))
{
}

torch::Tensor AdaBelief::step(LossClosure closure)
{
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    for (auto& group : param_groups_)
    {
        auto& options = static_cast<AdaBeliefOptions&>(group.options());

        for (auto& p : group.params())
        {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse())
            {
                throw std::runtime_error("AdaBelief does not support sparse gradients.");
            }

            auto& state_ptr = state_.at(p.unsafeGetTensorImpl());
            auto& state = static_cast<AdaBeliefParamState&>(*state_ptr.get());

            if (!state.step().defined())
            {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.exp_avg(torch::zeros_like(p, torch::MemoryFormat::Preserve));
                state.exp_avg_var(torch::zeros_like(p, torch::MemoryFormat::Preserve)); // s_t
            }

            auto& exp_avg = state.exp_avg();
            auto& exp_avg_var = state.exp_avg_var();

            torch::Tensor current_step_cpu = state.step();
            current_step_cpu += 1.0;
            state.step(current_step_cpu);
            double current_step_val = current_step_cpu.item<double>();

            // Apply weight decay
            if (options.weight_decay() != 0.0)
            {
                grad = grad.add(p.detach(), options.weight_decay());
            }

            double beta1 = options.beta1();
            double beta2 = options.beta2();

            // Update biased first moment estimate (m_t)
            exp_avg.mul_(beta1).add_(grad, 1.0 - beta1);

            // Update biased second moment estimate (s_t)
            // s_t = beta2 * s_{t-1} + (1 - beta2) * (g_t - m_t)^2
            // Need to calculate (g_t - m_t)
            // Note: exp_avg here is m_t (already updated for current step)
            torch::Tensor grad_minus_exp_avg = grad - exp_avg;
            exp_avg_var.mul_(beta2).addcmul_(grad_minus_exp_avg, grad_minus_exp_avg, 1.0 - beta2);
            // Add eps to s_t inside the sqrt to prevent division by zero,
            // AND to prevent s_t from becoming exactly zero if g_t is always equal to m_t.
            // The paper adds eps to s_t *before* bias correction.

            // Bias correction
            // Effective m_hat_t = m_t / (1 - beta1^t)
            double bias_correction1 = 1.0 - std::pow(beta1, current_step_val);
            // Effective s_hat_t = s_t / (1 - beta2^t)
            double bias_correction2 = 1.0 - std::pow(beta2, current_step_val);

            // Note: The original AdaBelief paper applies eps to s_t *before* bias correction,
            // then takes sqrt. Some implementations add eps *after* sqrt.
            // Let's follow a common pattern: add eps to s_hat_t inside the sqrt.
            // Or, more closely to paper: sqrt(s_t_hat + eps) where s_t_hat is bias corrected.
            // The paper actually does: sqrt(s_t / bias_correction2) + eps (eps outside sqrt)
            // OR sqrt(s_t / bias_correction2 + eps_inside_sqrt) (eps inside sqrt)
            // Let's use eps inside the sqrt for numerical stability, similar to Adam.

            // Denominator: sqrt(s_hat_t) + eps
            // s_hat_t = exp_avg_var / bias_correction2
            // denom = sqrt( (exp_avg_var / bias_correction2) + options.eps() )
            // No, the original paper is: (sqrt(exp_avg_var / bias_correction2) + options.eps())
            // Let's stick to the paper's formula (eps outside the sqrt initially, but then adjusted)
            // The key is s_t needs to be non-negative. (g-m)^2 ensures this.
            // AdaBelief formula is: m_hat / (sqrt(s_hat) + eps)
            // where s_hat is the bias-corrected exp_avg_var.

            // Let's use a formulation common in Adam-like optimizers for the denominator for stability:
            // denom = ( (exp_avg_var / bias_correction2).sqrt_() ).add_(options.eps());
            // No, the paper and official code add eps to s_hat *before* sqrt for AdaBelief:
            // S_hat_t = S_t / (1 - beta_2^t)
            // Update = m_hat_t / (sqrt(S_hat_t) + eps)
            // OR Update = m_hat_t / (sqrt(S_hat_t + eps_stabilizer))
            // The official PyTorch implementation seems to use:
            // denom = (exp_avg_var.sqrt() / math.sqrt(bias_correction2_t)).add_(group['eps'])
            // where exp_avg_var is s_t. This is Adam-like.
            //
            // For AdaBelief, it's actually S_hat = S_t / bias_correction2
            // Denom = sqrt(S_hat) + eps OR sqrt(S_hat + eps_for_sqrt)
            // The paper's Equation 4: denom = sqrt(S_hat_t) + epsilon
            // where S_hat_t = S_t / (1-beta2^t)
            // And S_t = beta2 * S_{t-1} + (1-beta2)*(g_t - m_t)^2 + epsilon_s_t
            // where epsilon_s_t is a small value added to (g_t-m_t)^2 to prevent it from being zero.
            // Let's simplify and add eps to the bias-corrected S_hat before sqrt for stability.

            torch::Tensor s_hat_t = exp_avg_var / bias_correction2;
            torch::Tensor denom = (s_hat_t + options.eps()).sqrt_();
            // Add eps before sqrt for stability, common variant.
            // The original paper has sqrt(s_hat) + eps_denom.
            // Using eps inside sqrt is more like AdamW/Adam.
            // Let's try the paper: sqrt(s_hat_t) and add eps AFTER.
            // denom = s_hat_t.sqrt().add_(options.eps()); // Paper's formulation for denominator part

            // Final Update:
            // Step size (alpha_t in paper) = lr / (1 - beta1^t)
            // This is incorrect. lr * m_hat / denom. m_hat = exp_avg / bias_correction1
            // So, (lr / bias_correction1) * exp_avg / denom
            // Or, lr * (exp_avg / bias_correction1) / denom

            // Bias corrected m_hat_t
            torch::Tensor m_hat_t = exp_avg / bias_correction1;

            // Parameter update
            // p_new = p_old - lr * m_hat_t / (sqrt(s_hat_t) + eps_denom)
            // We use options.lr() as the main learning rate. Bias correction for m_t handles the step size adjustment part.
            p.data().addcdiv_(m_hat_t, denom.add(options.eps()), -options.lr());
            // The above denom.add(options.eps()) is one way.
            // Let's refine based on typical AdaBelief code structure:
            // denom = (exp_avg_var / bias_correction2).sqrt().add_(options.eps()); // This is more Adam-like
            //
            // A common AdaBelief implementation is closer to:
            // s_hat = exp_avg_var / bias_correction2
            // update = (exp_avg / bias_correction1) / (s_hat.sqrt() + options.eps())
            // p.add_(update, -options.lr())
            // which is p.addcdiv_(exp_avg / bias_correction1, s_hat.sqrt().add_(options.eps()), -options.lr())

            // Let's use a clear formulation:
            // m_hat = exp_avg / bias_correction1
            // s_hat = exp_avg_var / bias_correction2
            // p.data_ += -lr * m_hat / (torch.sqrt(s_hat) + eps)

            // Final revised denominator based on common practice for AdaBelief:
            // Add a small epsilon to s_hat to prevent sqrt of zero if s_hat is zero,
            // then add the main eps for numerical stability against division by zero.
            // This is slightly different from adding eps only after sqrt.
            s_hat_t = exp_avg_var / bias_correction2;
            torch::Tensor final_denom = (s_hat_t + options.eps()).sqrt_().add_(options.eps());
            // Add eps inside and outside sqrt for robustness

            p.data().addcdiv_(m_hat_t, final_denom, -options.lr());
        }
    }
    return loss;
}

void AdaBelief::save(torch::serialize::OutputArchive& archive) const
{
    torch::optim::Optimizer::save(archive);
}

void AdaBelief::load(torch::serialize::InputArchive& archive)
{
    torch::optim::Optimizer::load(archive);
}

std::unique_ptr<torch::optim::OptimizerParamState> AdaBelief::make_param_state()
{
    return std::make_unique<AdaBeliefParamState>();
}
