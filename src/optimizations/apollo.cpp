#include "include/optimizations/apollo.h"
#include <torch/autograd.h> // For autograd::backward
#include <stdexcept>
namespace xt::optim
{
    // --- Apollo Implementation ---

    Apollo::Apollo(std::vector<torch::Tensor> params, ApolloOptions options)
        : params_(std::move(params)), options_(std::move(options)) {}

    void Apollo::_init_state(torch::Tensor& p) {
        if (state_.find(p.unsafeGetTensorImpl()) == state_.end()) {
            auto new_state = std::make_unique<ApolloParamState>();
            new_state->step(torch::tensor(0.0, torch::dtype(torch::kFloat64).device(torch::kCPU)));
            new_state->exp_avg(torch::zeros_like(p));
            new_state->hess_diag_ema(torch::zeros_like(p));
            new_state->bias_rectifier(torch::zeros_like(p)); // B_t starts at 0
            state_[p.unsafeGetTensorImpl()] = std::move(new_state);
        }
    }

    void Apollo::zero_grad() {
        for (auto& p : params_) {
            if (p.grad().defined()) {
                p.grad().detach_();
                p.grad().zero_();
            }
        }
    }

    void Apollo::step(torch::Tensor& loss) {
        TORCH_CHECK(torch::GradMode::is_enabled(), "Apollo::step() must be called with autograd enabled.");

        // --- First Backward Pass: Compute standard gradients dL/dw ---
        torch::autograd::backward({loss}, /*grad_tensors=*/{}, /*retain_graph=*/true);

        std::vector<torch::Tensor> standard_grads;
        std::vector<torch::Tensor> z_vectors;
        for (const auto& p : params_) {
            TORCH_CHECK(p.grad().defined(), "Apollo: All parameters must have a gradient.");
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

        torch::NoGradGuard no_grad;

        int i = 0;
        for (auto& p : params_) {
            _init_state(p);
            auto& state = *state_.at(p.unsafeGetTensorImpl());

            state.step(state.step() + 1.0);
            double current_step_val = state.step().item<double>();

            const auto& grad = standard_grads[i];
            const auto& hess_vec_prod = p.grad();
            const auto& z_rand = z_vectors[i];

            // 1. Update first moment (m_t) with the standard gradient
            auto& m = state.exp_avg();
            m.mul_(options_.beta1).add_(grad, 1.0 - options_.beta1);

            // 2. Update second moment (v_t) with the Hessian diagonal
            auto D_t = (hess_vec_prod * z_rand).abs();
            auto& v = state.hess_diag_ema();
            v.mul_(options_.beta2).add_(D_t, 1.0 - options_.beta2);

            // 3. Bias correction for both moments
            double bias_correction1 = 1.0 - std::pow(options_.beta1, current_step_val);
            double bias_correction2 = 1.0 - std::pow(options_.beta2, current_step_val);
            auto m_hat = m / bias_correction1;
            auto v_hat = v / bias_correction2;

            // 4. Update the bias rectifier B_t
            // B_t = max(B_{t-1}, v_hat_t)
            auto& B = state.bias_rectifier();
            torch::max_out(B, B, v_hat);

            // 5. Compute the update direction using the rectified denominator
            auto denom = B.sqrt().add(options_.eps);
            auto update_direction = m_hat / denom;

            // 6. Add decoupled weight decay to the direction
            if (options_.weight_decay > 0.0) {
                update_direction.add_(p.detach(), options_.weight_decay);
            }

            // 7. Compute LAMB-like trust ratio
            auto weight_norm = p.detach().norm(2).item<double>();
            auto update_norm = update_direction.norm(2).item<double>();

            double trust_ratio = 1.0;
            if (weight_norm > 0 && update_norm > 0) {
                trust_ratio = weight_norm / update_norm;
            }

            // 8. Final update
            p.data().add_(update_direction, -options_.lr * trust_ratio);

            i++;
        }
    }
}