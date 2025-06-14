#include "include/optimizations/ada_hessian.h"
#include <stdexcept>
#include <torch/autograd.h> // Include for torch::autograd::backward

// ... (Other methods like the constructor, zero_grad, _init_state are correct) ...

// --- AdaHessian Implementation ---

// CORRECTED step method
void AdaHessian::step(torch::Tensor& loss) {
    TORCH_CHECK(torch::GradMode::is_enabled(), "AdaHessian::step() must be called with autograd enabled.");

    // --- First Backward Pass: Compute standard gradients dL/dw ---
    // Use torch::autograd::backward to keep the graph alive for the second pass.
    torch::autograd::backward({loss}, /*grad_tensors=*/{}, /*retain_graph=*/true);

    // Store the standard gradients and create the random vector z
    std::vector<torch::Tensor> standard_grads;
    std::vector<torch::Tensor> z_vectors;
    for (const auto& p : params_) {
        TORCH_CHECK(p.grad().defined(), "AdaHessian: All parameters must have a gradient after the first backward pass.");
        standard_grads.push_back(p.grad().clone());

        // Generate Rademacher random vector z (+1 or -1)
        auto z = (torch::randint_like(p, 0, 2) * 2 - 1).to(p.dtype());
        z_vectors.push_back(z);
    }

    // --- Second Backward Pass: Compute H*z ---
    // Compute the dot product of the standard gradients and z
    torch::Tensor grad_z_dot_product = torch::tensor(0.0, loss.options());
    for(size_t i = 0; i < standard_grads.size(); ++i) {
        grad_z_dot_product += (standard_grads[i] * z_vectors[i]).sum();
    }

    // We need to zero out the old grads before the second pass.
    zero_grad();
    // The second backward pass computes d(grad_z_dot_product)/dw, which is H*z.
    // We don't need to retain the graph after this.
    grad_z_dot_product.backward();

    // Now, p.grad() contains the H*z vectors.

    torch::NoGradGuard no_grad; // From here on, we do standard optimizer updates.

    int i = 0;
    for (auto& p : params_) {
        // Ensure state is initialized
        _init_state(p);
        auto& state = *state_.at(p.unsafeGetTensorImpl());

        state.step(state.step() + 1.0);
        double current_step_val = state.step().item<double>();

        auto& m = state.exp_avg();
        auto& v = state.exp_avg_sq();

        // The original gradient from the first backward pass
        const auto& grad = standard_grads[i];
        // The H*z vector from the second backward pass
        const auto& hessian_vec_prod = p.grad(); // Currently holds H*z
        // The Rademacher vector
        const auto& z = z_vectors[i];

        // 1. Approximate Hessian diagonal: D_t = |H_t*z| * z
        auto hessian_diag_approx = (hessian_vec_prod * z).abs();

        // 2. Update Adam-like states
        m.mul_(options_.beta1).add_(grad, 1.0 - options_.beta1);
        v.mul_(options_.beta2).addcmul_(hessian_diag_approx, hessian_diag_approx, 1.0 - options_.beta2);

        // 3. Bias correction
        double bias_correction1 = 1.0 - std::pow(options_.beta1, current_step_val);
        double bias_correction2 = 1.0 - std::pow(options_.beta2, current_step_val);

        auto m_hat = m / bias_correction1;
        auto v_hat = v / bias_correction2;

        // 4. Decoupled Weight Decay & Final Update
        if (options_.weight_decay > 0.0) {
            p.data().add_(p.data(), -options_.lr * options_.weight_decay);
        }

        auto denom = v_hat.sqrt().add_(options_.eps);
        p.data().addcdiv_(m_hat, denom, -options_.lr);

        i++;
    }
}