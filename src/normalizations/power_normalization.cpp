#include "include/normalizations/power_normalization.h"





#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath> // For std::pow, std::abs, std::copysign

// Forward declaration for the Impl struct
struct PowerNormalizationImpl;

// The main module struct that users will interact with.
struct PowerNormalization : torch::nn::ModuleHolder<PowerNormalizationImpl> {
    using torch::nn::ModuleHolder<PowerNormalizationImpl>::ModuleHolder;

    torch::Tensor forward(torch::Tensor x) {
        return impl_->forward(x);
    }
};

// The implementation struct for PowerNormalization
struct PowerNormalizationImpl : torch::nn::Module {
    double power_val_;      // The exponent 'p'
    bool learnable_power_;  // Whether 'p' is learnable
    bool apply_l2_norm_;    // Whether to apply L2 norm before power transform
    bool signed_power_;     // If true, use sgn(x) * |x|^p, else x^p (risks NaN for x<0, non-integer p)
    double eps_l2_;         // Epsilon for L2 normalization (if applied)
    int64_t l2_norm_dim_;   // Dimension for L2 norm (-1 for global, or specific axis)

    // Learnable parameter for power (if learnable_power_ is true)
    torch::Tensor power_param_;

    PowerNormalizationImpl(double initial_power = 0.5, // e.g., 0.5 for sqrt normalization
                           bool learnable_power = false,
                           bool apply_l2_norm = false,
                           int64_t l2_norm_dim = -1, // -1 for global L2 norm
                           bool signed_power = true,
                           double eps_l2 = 1e-8)
        : power_val_(initial_power),
          learnable_power_(learnable_power),
          apply_l2_norm_(apply_l2_norm),
          l2_norm_dim_(l2_norm_dim),
          signed_power_(signed_power),
          eps_l2_(eps_l2) {

        if (learnable_power_) {
            // Register power_param_ and initialize it.
            // The actual power used will be derived from this (e.g., just power_param_ itself, or softplus(power_param_) to keep it positive).
            // For simplicity, let power_param_ directly be the power.
            power_param_ = register_parameter("power_exponent", torch::tensor({initial_power}));
        }
        // If not learnable, power_val_ is used directly.
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: input tensor

        torch::Tensor x_processed = x;

        // --- 1. Optional L2 Normalization ---
        if (apply_l2_norm_) {
            torch::Tensor norm;
            if (l2_norm_dim_ == -1) { // Global L2 norm
                norm = x_processed.norm();
            } else { // L2 norm along a specific dimension
                TORCH_CHECK(x_processed.dim() > l2_norm_dim_ && l2_norm_dim_ >= 0,
                            "Input tensor rank must be greater than l2_norm_dim, and l2_norm_dim must be non-negative or -1.");
                norm = x_processed.norm(2, l2_norm_dim_, /*keepdim=*/true);
            }
            x_processed = x_processed / (norm + eps_l2_);
        }

        // --- 2. Power Transformation ---
        double current_power = learnable_power_ ? power_param_.item<double>() : power_val_;

        torch::Tensor output;
        if (signed_power_) {
            // output = sgn(x) * |x|^p
            // torch::sgn and torch::pow handle this well.
            // torch::abs(x_processed) might be needed if power can be non-integer and x_processed can be negative.
            // pow(negative_base, non_integer_exponent) is complex or NaN.
            // sgn(x) * (|x| + eps_for_pow)^p to avoid pow(0, p<1) issues if needed, but torch.pow should handle.
            output = torch::sgn(x_processed) * torch::pow(torch::abs(x_processed), current_power);
        } else {
            // output = x^p
            // This can result in NaN if x contains negative values and current_power is not an integer.
            // A common use case for x^p is when x is known to be non-negative (e.g., after ReLU).
            if ((x_processed < 0).any().item<bool>() && (current_power != std::round(current_power) || current_power < 0)) {
                 TORCH_WARN_ONCE("Applying non-signed power with potentially negative inputs and non-integer/negative exponent. This may result in NaNs.");
            }
            output = torch::pow(x_processed, current_power);
        }

        return output;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "PowerNormalization(initial_power=" << (learnable_power_ ? power_param_.item<double>() : power_val_)
               << ", learnable_power=" << (learnable_power_ ? "true" : "false")
               << ", apply_l2_norm=" << (apply_l2_norm_ ? "true" : "false");
        if (apply_l2_norm_) {
            stream << ", l2_norm_dim=" << l2_norm_dim_ << ", eps_l2=" << eps_l2_;
        }
        stream << ", signed_power=" << (signed_power_ ? "true" : "false") << ")";
    }
};
TORCH_MODULE(PowerNormalization);


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    // --- Test Case 1: Square-root normalization (power=0.5), signed ---
    std::cout << "--- Test Case 1: Signed Sqrt Normalization (power=0.5) ---" << std::endl;
    PowerNormalization pownorm_module1(0.5, false, false, -1, true);
    // std::cout << pownorm_module1 << std::endl;

    torch::Tensor x1 = torch::tensor({-4.0, -1.0, 0.0, 1.0, 4.0, 9.0}); // Mixed signs
    std::cout << "Input x1: " << x1 << std::endl;

    torch::Tensor y1 = pownorm_module1->forward(x1);
    std::cout << "Output y1 (signed sqrt): " << y1 << std::endl;
    // Expected: {-2.0, -1.0, 0.0, 1.0, 2.0, 3.0}
    torch::Tensor expected_y1 = torch::tensor({-2.0, -1.0, 0.0, 1.0, 2.0, 3.0});
    TORCH_CHECK(torch::allclose(y1, expected_y1), "Signed sqrt normalization failed.");


    // --- Test Case 2: L2 norm then power transform (power=2.0, unsigned) ---
    std::cout << "\n--- Test Case 2: L2 norm then Unsigned Power (power=2.0) ---" << std::endl;
    int N2=1, C2=3, H2=1, W2=1; // Treat as a single 3-dim vector for global L2 norm
    PowerNormalization pownorm_module2(2.0, false, true, -1, false); // apply_l2_norm=true, l2_norm_dim=-1 (global), signed_power=false
    // std::cout << pownorm_module2 << std::endl;

    torch::Tensor x2 = torch::tensor({{{{{3.0, 4.0, 0.0}}}}}); // Shape (1,1,1,1,3) -> will be (1,3,1,1) if used as NCHW
    x2 = x2.reshape({N2, C2, H2, W2}); // (1, 3, 1, 1)
    std::cout << "Input x2: " << x2.flatten() << std::endl;
    // L2 norm of (3,4,0) is sqrt(9+16+0) = 5.
    // L2 normalized x2_l2 = (3/5, 4/5, 0/5) = (0.6, 0.8, 0.0)
    // (x2_l2)^2 = (0.36, 0.64, 0.0)

    torch::Tensor y2 = pownorm_module2->forward(x2);
    std::cout << "Output y2 (L2 norm then square): " << y2.flatten() << std::endl;
    torch::Tensor expected_y2_flat = torch::tensor({0.36, 0.64, 0.0});
    TORCH_CHECK(torch::allclose(y2.flatten(), expected_y2_flat, 1e-5), "L2 norm then square failed.");


    // --- Test Case 3: Learnable power, signed ---
    std::cout << "\n--- Test Case 3: Learnable power, signed ---" << std::endl;
    PowerNormalization pownorm_module3(0.7, true, false, -1, true); // initial power 0.7, learnable
    pownorm_module3->train();
    // std::cout << pownorm_module3 << std::endl;
    std::cout << "Initial learnable power: " << pownorm_module3->power_param_.item<double>() << std::endl;

    torch::Tensor x3 = torch::tensor({-8.0, 27.0}, torch::requires_grad());
    torch::Tensor initial_y3 = pownorm_module3->forward(x3.detach()); // Forward with initial power
    std::cout << "Output with initial power (0.7): " << initial_y3 << std::endl;

    // Simulate a backward pass and parameter update (simplified)
    torch::optim::SGD optimizer({pownorm_module3->power_param_}, 0.1);
    optimizer.zero_grad();
    torch::Tensor y3_for_loss = pownorm_module3->forward(x3);
    torch::Tensor loss = y3_for_loss.sum(); // Dummy loss
    loss.backward();
    optimizer.step();

    std::cout << "Updated learnable power: " << pownorm_module3->power_param_.item<double>() << std::endl;
    TORCH_CHECK(pownorm_module3->power_param_.item<double>() != 0.7, "Learnable power did not update.");
    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    TORCH_CHECK(grad_exists_x3, "No gradient for x3!");


    // --- Test Case 4: Unsigned power with negative input (should warn) ---
    std::cout << "\n--- Test Case 4: Unsigned power with negative input (power=0.5) ---" << std::endl;
    PowerNormalization pownorm_module4(0.5, false, false, -1, false); // signed_power = false
    torch::Tensor x4 = torch::tensor({-4.0, 4.0});
    std::cout << "Input x4: " << x4 << std::endl;
    std::cout << "Executing forward for Test Case 4 (expect a warning about NaNs)..." << std::endl;
    torch::Tensor y4 = pownorm_module4->forward(x4);
    std::cout << "Output y4 (unsigned power 0.5): " << y4 << std::endl; // First element will be NaN
    TORCH_CHECK(torch::isnan(y4[0]).item<bool>(), "Expected NaN for unsigned sqrt of negative number.");
    TORCH_CHECK(std::abs(y4[1].item<double>() - 2.0) < 1e-6, "Positive part of unsigned sqrt failed.");


    std::cout << "\nPowerNormalization tests finished." << std::endl;
    return 0;
}


namespace xt::norm
{
    auto PowerNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
