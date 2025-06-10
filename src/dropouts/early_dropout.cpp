#include "include/dropouts/early_dropout.h"


#include <torch/torch.h>
#include <ostream> // For std::ostream

struct EasyDropoutImpl : torch::nn::Module {
    double p_drop_; // Probability of an element to be zeroed out.
    double epsilon_ = 1e-7; // For numerical stability in division

    EasyDropoutImpl(double p_drop = 0.5) : p_drop_(p_drop) {
        TORCH_CHECK(p_drop_ >= 0.0 && p_drop_ <= 1.0, "Dropout probability p_drop must be between 0 and 1.");
    }

    torch::Tensor forward(const torch::Tensor& input) {
        if (!this->is_training() || p_drop_ == 0.0) {
            // If not in training mode or if dropout probability is zero,
            // return the input as is.
            return input;
        }

        if (p_drop_ == 1.0) {
            // If dropout probability is one, all elements are zeroed out.
            return torch::zeros_like(input);
        }

        // Calculate keep probability.
        double keep_prob = 1.0 - p_drop_;

        // Create a binary mask:
        // Elements are 1 with probability 'keep_prob' (kept)
        // Elements are 0 with probability 'p_drop_' (dropped)
        // The mask has the same shape and device as the input.
        torch::Tensor mask = torch::bernoulli(
            torch::full_like(input, keep_prob)
        ).to(input.dtype());

        // Apply the mask: zero out elements where mask is 0.
        // Scale the remaining elements by 1/keep_prob (inverted dropout).
        // This scaling ensures that the expected sum of outputs remains the same
        // as the sum of inputs, which means no changes are needed during inference.
        return (input * mask) / (keep_prob + epsilon_);
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "EasyDropout(p_drop=" << p_drop_ << ")";
    }
};

TORCH_MODULE(EasyDropout); // Creates the EasyDropout module "class"

/*
// Example of how to use the EasyDropout module:
// (This is for illustration and would typically be in your main application code)

#include <iostream>

void run_easy_dropout_example() {
    torch::manual_seed(0); // For reproducible results

    double dropout_rate = 0.3; // 30% chance of an element being zeroed
    EasyDropout dropout_module(dropout_rate);
    std::cout << "EasyDropout Module: " << dropout_module << std::endl;

    // Example input tensor
    torch::Tensor input_tensor = torch::ones({2, 5}); // Batch_size=2, Features=5
    std::cout << "Input Tensor (all ones):\n" << input_tensor << std::endl;

    // --- Training mode ---
    dropout_module->train(); // Set the module to training mode
    torch::Tensor output_train = dropout_module->forward(input_tensor);
    std::cout << "Output (training mode, p_drop=" << dropout_rate << "):\n" << output_train << std::endl;
    // Expected: Approximately 30% of elements will be zero.
    // Non-zero elements will be scaled by 1 / (1 - 0.3) = 1 / 0.7 approx 1.428.
    // So, non-zero elements will be around 1.428.

    // --- Evaluation mode ---
    dropout_module->eval(); // Set the module to evaluation mode
    torch::Tensor output_eval = dropout_module->forward(input_tensor);
    std::cout << "Output (evaluation mode):\n" << output_eval << std::endl;
    // Expected: Output should be identical to the input tensor in evaluation mode.
    TORCH_CHECK(torch::allclose(input_tensor, output_eval), "EasyDropout eval output mismatch!");


    // --- Test with p_drop = 0.0 (no dropout) ---
    EasyDropout no_dropout_module(0.0);
    no_dropout_module->train();
    torch::Tensor output_no_drop = no_dropout_module->forward(input_tensor);
    std::cout << "\nOutput (training mode, p_drop=0.0):\n" << output_no_drop << std::endl;
    TORCH_CHECK(torch::allclose(input_tensor, output_no_drop), "EasyDropout p_drop=0.0 output mismatch!");


    // --- Test with p_drop = 1.0 (drop everything) ---
    EasyDropout full_dropout_module(1.0);
    full_dropout_module->train();
    torch::Tensor output_full_drop = full_dropout_module->forward(input_tensor);
    std::cout << "\nOutput (training mode, p_drop=1.0):\n" << output_full_drop << std::endl;
    TORCH_CHECK(torch::allclose(torch::zeros_like(input_tensor), output_full_drop), "EasyDropout p_drop=1.0 output mismatch!");

}

// int main() {
//    run_easy_dropout_example();
//    return 0;
// }
*/

namespace xt::dropouts
{
    torch::Tensor early_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto EarlyDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::early_dropout(torch::zeros(10));
    }
}
