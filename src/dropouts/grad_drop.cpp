#include "include/dropouts/grad_drop.h"


#include <torch/torch.h>
#include <cmath>   // For std::log, std::abs
#include <algorithm> // For std::clamp
#include <ostream> // For std::ostream

namespace { // Anonymous namespace for helper utility
// Calculates logit(p) = log(p / (1-p)) with clamping for numerical stability.
double calculate_logit(double p, double epsilon = 1e-7) {
    p = std::clamp(p, epsilon, 1.0 - epsilon);
    return std::log(p / (1.0 - p));
}
} // namespace

struct GradDropImpl : torch::nn::Module {
    double p_drop_at_zero_activation_; // Dropout probability if activation is zero
    double sensitivity_;               // How fast keep_prob increases with abs(activation)
    double base_logit_keep_at_zero_;   // Precomputed logit(1 - p_drop_at_zero_activation_)
    double epsilon_ = 1e-7;           // For numerical stability

    GradDropImpl(double p_drop_at_zero_activation = 0.5, double sensitivity = 1.0)
        : p_drop_at_zero_activation_(p_drop_at_zero_activation),
          sensitivity_(sensitivity) {
        TORCH_CHECK(p_drop_at_zero_activation_ >= 0.0 && p_drop_at_zero_activation_ < 1.0,
                    "p_drop_at_zero_activation must be in [0, 1).");
        TORCH_CHECK(sensitivity_ >= 0.0, "sensitivity must be non-negative.");

        base_logit_keep_at_zero_ = calculate_logit(1.0 - p_drop_at_zero_activation_, epsilon_);
    }

    torch::Tensor forward(const torch::Tensor& input) {
        if (!this->is_training()) {
            return input;
        }
        // If p_drop_at_zero is 0 and sensitivity is 0, it means keep_prob is always logit_keep(1.0)=inf -> sigmoid=1 (no drop)
        // This case is effectively no dropout if params are set to imply full keep.
        // A specific check for p_drop_at_zero_ == 0 && sensitivity_ == 0.0 might be redundant
        // as sigmoid(very_large_number) is 1.0.

        // Calculate element-wise keep probability logits
        torch::Tensor abs_input = torch::abs(input);
        torch::Tensor keep_prob_logits = base_logit_keep_at_zero_ + sensitivity_ * abs_input;

        // Calculate element-wise keep probabilities
        torch::Tensor keep_probabilities = torch::sigmoid(keep_prob_logits);

        // Clamp keep_probabilities to avoid division by zero or extreme scaling.
        // Esp. if keep_prob can be very close to 0 for some units.
        torch::Tensor keep_probabilities_clamped = torch::clamp(keep_probabilities, epsilon_, 1.0);
        // If a keep_probability is 1.0, then that unit is never dropped by this mechanism.
        // If keep_probability is epsilon_, it's almost always dropped.

        // Generate mask based on these element-wise keep probabilities
        torch::Tensor mask = torch::bernoulli(keep_probabilities).to(input.dtype());

        // Apply mask and scale (inverted dropout, element-wise scaling)
        return (input * mask) / keep_probabilities_clamped;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "GradDrop(p_drop_at_zero_activation=" << p_drop_at_zero_activation_
               << ", sensitivity=" << sensitivity_
               << ", base_logit_keep_at_zero=" << base_logit_keep_at_zero_ << ")";
    }
};

TORCH_MODULE(GradDrop); // Creates the GradDrop module "class"

/*
// Example of how to use the GradDrop module:
// (This is for illustration and would typically be in your main application code)

#include <iostream>

void run_grad_drop_example() {
    torch::manual_seed(0); // For reproducible results

    // Scenario 1: Moderate dropout for zero activations, moderate sensitivity
    GradDrop dropout_module1(0.5, 1.0); // p_drop_at_zero=0.5, sensitivity=1.0
                                        // base_logit_keep_at_zero = logit(0.5) = 0.0
    std::cout << "GradDrop Module 1: " << dropout_module1 << std::endl;

    torch::Tensor input_tensor = torch::tensor({{-2.0, -0.1, 0.0, 0.1, 2.0, 5.0}}, torch::kFloat32);
    // Activations:                          -2.0, -0.1,  0.0,  0.1,  2.0, 5.0
    // Abs Activations:                       2.0,  0.1,  0.0,  0.1,  2.0, 5.0
    // Keep Logits (0.0 + 1.0 * abs_act):     2.0,  0.1,  0.0,  0.1,  2.0, 5.0
    // Keep Probs (sigmoid(logit)):         0.88, 0.52, 0.50, 0.52, 0.88, 0.99 (approx)

    std::cout << "Input Tensor:\n" << input_tensor << std::endl;

    dropout_module1->train(); // Set to training mode
    torch::Tensor output_train1 = dropout_module1->forward(input_tensor);
    std::cout << "Output (train, module1):\n" << output_train1 << std::endl;
    // For input 0.0, keep_prob is ~0.5. For input 2.0, keep_prob is ~0.88. Higher chance of keeping larger activations.


    // Scenario 2: High dropout for zero activations, low sensitivity
    GradDrop dropout_module2(0.8, 0.1); // p_drop_at_zero=0.8, sensitivity=0.1
                                        // base_logit_keep_at_zero = logit(0.2) approx -1.386
    std::cout << "\nGradDrop Module 2: " << dropout_module2 << std::endl;
    // Activations:                          -2.0,  -0.1,   0.0,   0.1,   2.0,  5.0
    // Abs Activations:                       2.0,   0.1,   0.0,   0.1,   2.0,  5.0
    // Keep Logits (-1.386 + 0.1 * abs_act):-1.186,-1.376,-1.386,-1.376,-1.186,-0.886
    // Keep Probs (sigmoid(logit)):         0.23,  0.20,  0.20,  0.20,  0.23,  0.30 (approx)
    // Low keep probabilities overall, slowly increasing.

    dropout_module2->train();
    torch::Tensor output_train2 = dropout_module2->forward(input_tensor);
    std::cout << "Output (train, module2):\n" << output_train2 << std::endl;

    // --- Evaluation mode ---
    dropout_module1->eval(); // Set to evaluation mode
    torch::Tensor output_eval = dropout_module1->forward(input_tensor);
    std::cout << "\nOutput (evaluation mode, module1):\n" << output_eval << std::endl;
    // Expected: Output should be identical to the input tensor in evaluation mode.
    TORCH_CHECK(torch::allclose(input_tensor, output_eval), "GradDrop eval output mismatch!");


    // Scenario 3: Low dropout for zero activations, high sensitivity
    GradDrop dropout_module3(0.1, 5.0); // p_drop_at_zero=0.1, sensitivity=5.0
                                        // base_logit_keep_at_zero = logit(0.9) approx 2.197
    std::cout << "\nGradDrop Module 3: " << dropout_module3 << std::endl;
    // Activations:                          -2.0, -0.1,  0.0,   0.1,   2.0,  5.0
    // Abs Activations:                       2.0,  0.1,  0.0,   0.1,   2.0,  5.0
    // Keep Logits (2.197 + 5.0 * abs_act): 12.197, 2.697,2.197, 2.697,12.197,27.197
    // Keep Probs (sigmoid(logit)):        ~1.0,   0.93, 0.90,  0.93,  ~1.0, ~1.0 (approx)
    // High keep probabilities, very quickly approaching 1.0.

    dropout_module3->train();
    torch::Tensor output_train3 = dropout_module3->forward(input_tensor);
    std::cout << "Output (train, module3):\n" << output_train3 << std::endl;
}

// int main() {
//    run_grad_drop_example();
//    return 0;
// }
*/


namespace xt::dropouts
{
    torch::Tensor grad_drop(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto GradDrop::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::grad_drop(torch::zeros(10));
    }
}
