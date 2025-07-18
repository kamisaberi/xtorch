#include <optimizations/adopt.h>
#include <torch/autograd.h> // For autograd::backward
#include <stdexcept>

namespace xt::optim
{
    // --- AdOpt Implementation ---

    AdOpt::AdOpt(std::vector<torch::Tensor> params, AdOptOptions options)
        : params_(std::move(params)), options_(std::move(options)) {}

    void AdOpt::_init_state(torch::Tensor& p) {
        if (state_.find(p.unsafeGetTensorImpl()) == state_.end()) {
            auto new_state = std::make_unique<AdOptParamState>();
            new_state->step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
            new_state->z(torch::zeros_like(p));
            new_state->hess_diag_ema(torch::zeros_like(p));
            new_state->grad_ema(torch::zeros_like(p));
            new_state->grad_sq_ema(torch::zeros_like(p));
            state_[p.unsafeGetTensorImpl()] = std::move(new_state);
        }
    }

    void AdOpt::zero_grad() {
        for (auto& p : params_) {
            if (p.grad().defined()) {
                p.grad().detach_();
                p.grad().zero_();
            }
        }
    }

    void AdOpt::step(torch::Tensor& loss) {
        TORCH_CHECK(torch::GradMode::is_enabled(), "AdOpt::step() must be called with autograd enabled.");

        // --- First Backward Pass: Compute standard gradients dL/dw ---
        torch::autograd::backward({loss}, /*grad_tensors=*/{}, /*retain_graph=*/true);

        std::vector<torch::Tensor> standard_grads;
        std::vector<torch::Tensor> z_vectors;
        for (const auto& p : params_) {
            TORCH_CHECK(p.grad().defined(), "AdOpt: All parameters must have a gradient.");
            standard_grads.push_back(p.grad().clone());
            auto z = (torch::randint_like(p, 0, 2) * 2 - 1).to(p.dtype());
            z_vectors.push_back(z);
        }

        // --- Second Backward Pass: Compute H*z ---
        torch::Tensor grad_z_dot_product = torch::tensor(0.0, loss.options());
        for(size_t i = 0; i < standard_grads.size(); ++i) {
            grad_z_dot_product += (standard_grads[i] * z_vectors[i]).sum();
        }
        zero_grad();
        grad_z_dot_product.backward();

        // --- Main AdOpt Update Logic ---
        torch::NoGradGuard no_grad;

        int i = 0;
        for (auto& p : params_) {
            _init_state(p);
            auto& state = *state_.at(p.unsafeGetTensorImpl());

            state.step(state.step() + 1.0);

            const auto& g_t = standard_grads[i];
            const auto& hess_vec_prod = p.grad();
            const auto& z_rand = z_vectors[i];

            // 1. Update smoothed statistics
            auto& H_ema = state.hess_diag_ema();
            auto& g_ema = state.grad_ema();
            auto& g_sq_ema = state.grad_sq_ema();

            auto D_t = (hess_vec_prod * z_rand).abs(); // Approx Hessian diagonal

            H_ema.mul_(options_.beta2).add_(D_t, 1.0 - options_.beta2);
            g_ema.mul_(options_.beta1).add_(g_t, 1.0 - options_.beta1);
            g_sq_ema.mul_(options_.beta2).addcmul_(g_t, g_t, 1.0 - options_.beta2);

            // 2. Compute optimal, adaptive hyperparameters
            // These are simplified versions from the paper for a practical implementation.
            // L (smoothness) is approximated by ||H_ema||
            // mu (strong convexity) is approximated by ||g_ema||^2 / ||g_sq_ema||
            // sigma^2 (variance) is approximated by ||g_sq_ema|| - ||g_ema||^2

            auto L = H_ema.max().item<double>() + options_.eps; // Smoothness
            auto g_ema_norm_sq = g_ema.square().sum().item<double>();
            auto g_sq_ema_norm = g_sq_ema.sum().item<double>();

            // sigma_sq must be non-negative
            double sigma_sq = std::max(0.0, g_sq_ema_norm - g_ema_norm_sq);

            // Simplified rule for learning rate eta_t and momentum beta_t
            double eta_t = 1.0 / (L + options_.eps);
            double beta_t = 1.0 - std::sqrt(sigma_sq) * std::sqrt(eta_t) / (std::sqrt(g_ema_norm_sq) + options_.eps);
            beta_t = std::max(0.0, std::min(1.0 - options_.eps, beta_t)); // Clamp beta

            // 3. Update the dual variable z
            auto& z = state.z();
            z.mul_(beta_t).sub_(g_t, eta_t);

            // 4. Update the primal parameters w
            // w_{t+1} = -z_{t+1} / H_ema
            auto denom = H_ema + options_.eps;

            // Apply decoupled weight decay first
            if (options_.weight_decay > 0.0) {
                p.data().add_(p.data(), -options_.lr * options_.weight_decay);
            }

            p.data().copy_(-z / denom);

            i++;
        }
    }
}