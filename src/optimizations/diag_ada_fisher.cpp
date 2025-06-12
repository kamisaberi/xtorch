#include "include/optimizations/diag_ada_fisher.h"
#include <stdexcept> // For std::runtime_error

// --- DiagonalAdaFisher Class Implementation ---

DiagonalAdaFisher::DiagonalAdaFisher(std::vector<torch::Tensor> params, DiagonalAdaFisherOptions options)
    : torch::optim::Optimizer(std::move(params), std::make_unique<DiagonalAdaFisherOptions>(options))
{
}

DiagonalAdaFisher::DiagonalAdaFisher(std::vector<torch::Tensor> params, double lr_val)
    : DiagonalAdaFisher(std::move(params), DiagonalAdaFisherOptions(lr_val))
{
}

torch::Tensor DiagonalAdaFisher::step(LossClosure closure)
{
    torch::NoGradGuard no_grad;
    torch::Tensor loss = {};
    if (closure) { loss = closure(); }

    for (auto& group : param_groups_)
    {
        auto& options = static_cast<DiagonalAdaFisherOptions&>(group.options());

        for (auto& p : group.params())
        {
            if (!p.grad().defined()) { continue; }

            auto grad = p.grad();
            if (grad.is_sparse())
            {
                throw std::runtime_error("DiagonalAdaFisher does not support sparse gradients.");
            }

            auto& state_ptr = state_.at(p.unsafeGetTensorImpl());
            auto& state = static_cast<DiagonalAdaFisherParamState&>(*state_ptr.get());

            if (!state.step().defined())
            {
                state.step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
                state.fisher_diag_ema(torch::zeros_like(p, torch::MemoryFormat::Preserve));
            }

            auto& fisher_diag_ema = state.fisher_diag_ema();

            torch::Tensor current_step_cpu = state.step();
            current_step_cpu += 1.0;
            state.step(current_step_cpu);
            // double current_step_val = current_step_cpu.item<double>(); // For bias correction if used

            // Apply weight decay (L2 regularization)
            // Weight decay can be applied before or after scaling by Fisher info.
            // Applying before is common for AdamW-like behavior.
            // Here, we apply it to the gradient directly.
            torch::Tensor grad_final = grad;
            if (options.weight_decay() != 0.0)
            {
                // For Fisher/Natural Gradient, WD is often added to the FIM diagonal.
                // Or, as in AdamW, subtract wd * lr * param from param later.
                // For simplicity here, like Adam:
                grad_final = grad_final.add(p.detach(), options.weight_decay());
            }

            double beta = options.beta();

            // Update EMA of squared gradients (diagonal Fisher estimate)
            // F_t = beta * F_{t-1} + (1 - beta) * g_t^2
            fisher_diag_ema.mul_(beta).addcmul_(grad, grad, 1.0 - beta); // Using original grad for Fisher est.

            // Optional: Bias correction for fisher_diag_ema
            // This would make it more like Adam's v_t if current_step_val is used.
            // For pure RMSProp-like behavior, bias correction is often omitted for this term.
            // torch::Tensor fisher_diag_corrected = fisher_diag_ema;
            // if (options.bias_correction_fisher()) { // If an option was added
            //    double bias_correction_fisher = 1.0 - std::pow(beta, current_step_val);
            //    fisher_diag_corrected = fisher_diag_ema / bias_correction_fisher;
            // }


            // Denominator for preconditioning: sqrt(F_t) + eps
            // Or sqrt(F_t + eps_inside_sqrt) + eps_outside_sqrt
            // Let's use the common RMSProp/Adam style: sqrt(F_t) + eps
            torch::Tensor denom = fisher_diag_ema.sqrt().add_(options.eps());

            // Parameter update: p_new = p_old - lr * (grad_final / denom)
            p.data().addcdiv_(grad_final, denom, -options.lr());
        }
    }
    return loss;
}

void DiagonalAdaFisher::save(torch::serialize::OutputArchive& archive) const
{
    torch::optim::Optimizer::save(archive);
}

void DiagonalAdaFisher::load(torch::serialize::InputArchive& archive)
{
    torch::optim::Optimizer::load(archive);
}

std::unique_ptr<torch::optim::OptimizerParamState> DiagonalAdaFisher::make_param_state()
{
    return std::make_unique<DiagonalAdaFisherParamState>();
}
