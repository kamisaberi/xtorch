#include "include/dropouts/concrete_dropout.h"


#include <torch/torch.h>
#include <vector>
#include <cmath>     // For std::log
#include <ostream>   // For std::ostream

// Paper Reference:
// Yarin Gal, Jiri Hron, Alex Kendall. "Concrete Dropout." NeurIPS 2017.
// Link: https://arxiv.org/abs/1705.07832

namespace { // Anonymous namespace for helper utility
// Calculates initial log_alpha from an initial dropout rate p.
// log_alpha = log(p / (1-p))
double calculate_initial_log_alpha_value(double initial_dropout_rate, double epsilon = 1e-7) {
    if (initial_dropout_rate < epsilon) {
        initial_dropout_rate = epsilon;
    }
    if (initial_dropout_rate > 1.0 - epsilon) {
        initial_dropout_rate = 1.0 - epsilon;
    }
    return std::log(initial_dropout_rate / (1.0 - initial_dropout_rate));
}
} // namespace

struct ConcreteDropoutImpl : torch::nn::Module {
    torch::Tensor log_alpha_; // Unconstrained learnable parameter(s) for dropout probability
    double temperature_;
    double dropout_regularizer_factor_; // Factor to scale the p-based regularization term
    double epsilon_ = 1e-7; // Small constant for numerical stability

    ConcreteDropoutImpl(
        c10::IntArrayRef probability_shape = {}, // Shape of log_alpha (empty for scalar)
        double initial_dropout_rate = 0.05,      // Desired initial dropout probability p
        double temperature = 0.1,                // Temperature for Concrete distribution
        double dropout_regularizer = 1e-5)       // Multiplier for the regularization term
        : temperature_(temperature), dropout_regularizer_factor_(dropout_regularizer) {

        TORCH_CHECK(temperature_ > 0, "Temperature for Concrete Dropout must be positive.");

        double initial_log_alpha_val = calculate_initial_log_alpha_value(initial_dropout_rate, epsilon_);

        torch::Tensor log_alpha_init;
        if (probability_shape.empty()) { // Scalar probability
            log_alpha_init = torch::tensor(initial_log_alpha_val, torch::kFloat32);
        } else {
            // Create a tensor of the given shape, filled with the initial log_alpha value.
            log_alpha_init = torch::full(probability_shape, initial_log_alpha_val, torch::kFloat32);
        }
        log_alpha_ = register_parameter("log_alpha", log_alpha_init);
    }

    torch::Tensor forward(const torch::Tensor& input) {
        if (!this->is_training()) {
            // During evaluation, Concrete Dropout (typically) acts as an identity function.
            // The learned dropout probabilities are expected to have regularized the network weights.
            return input;
        }

        // Calculate dropout probability p from log_alpha.
        torch::Tensor p = torch::sigmoid(log_alpha_);

        // Clamp p for numerical stability in subsequent log operations.
        p = torch::clamp(p, epsilon_, 1.0 - epsilon_);

        // Handle broadcasting of p (e.g., for per-channel dropout).
        torch::Tensor p_broadcastable = p;
        if (p.dim() == 1 && input.dim() > 1 && p.size(0) == input.size(1) && input.size(1) > 0) {
            // Assuming p.size(0) is the channel dimension (dim 1 of input)
            std::vector<int64_t> view_shape(input.dim(), 1L);
            view_shape[1] = p.size(0); // e.g., [1, C, 1, 1] for NCHW input
            p_broadcastable = p.view(view_shape);
        }
        // Else: rely on PyTorch's standard broadcasting rules if p is scalar or already matches input.

        // Sample u ~ Uniform(epsilon, 1-epsilon) for numerical stability in logs.
        // The shape of unif_noise should match p_broadcastable to generate a mask of that shape.
        torch::Tensor unif_noise = torch::rand(p_broadcastable.sizes(), input.options());
        unif_noise = torch::clamp(unif_noise, epsilon_, 1.0 - epsilon_);

        // Calculate logits for the Concrete distribution:
        // logit_concrete = (logit(p) + logit(u)) / temperature
        torch::Tensor logit_p = torch::log(p_broadcastable) - torch::log(1.0 - p_broadcastable);
        torch::Tensor logit_unif = torch::log(unif_noise) - torch::log(1.0 - unif_noise);

        torch::Tensor concrete_logits = (logit_p + logit_unif) / temperature_;
        torch::Tensor concrete_mask = torch::sigmoid(concrete_logits); // This is the continuous mask z ~ Concrete(p, temp)

        // Apply the continuous mask.
        // Note: Unlike standard inverted dropout, scaling by 1/keep_prob is not typically done here.
        // The regularization term and learning process adjust for this.
        torch::Tensor output = input * concrete_mask;

        return output;
    }

    // Calculates the regularization term associated with the dropout probabilities 'p'.
    // This term should be ADDED to the main loss function during training.
    // The term is sum_units (p_unit * log(p_unit) + (1-p_unit) * log(1-p_unit)),
    // scaled by dropout_regularizer_factor_. This is related to the negative entropy
    // of Bernoulli(p) and encourages p to move towards 0 or 1.
    torch::Tensor calculate_regularization_term() const {
        torch::Tensor p = torch::sigmoid(log_alpha_);
        p = torch::clamp(p, epsilon_, 1.0 - epsilon_); // Ensure stability for log operations

        // Calculate p*log(p) + (1-p)*log(1-p) for each dropout probability.
        // This is the negative entropy of a Bernoulli distribution with parameter p.
        torch::Tensor neg_entropy_term = p * torch::log(p) + (1.0 - p) * torch::log(1.0 - p);

        // Sum this term over all dropout units (if p is a tensor, otherwise it's scalar already).
        // Then scale by the dropout_regularizer_factor_.
        return dropout_regularizer_factor_ * torch::sum(neg_entropy_term);
    }

    // Helper method to get the current dropout probabilities (sigmoid of log_alpha).
    torch::Tensor get_dropout_probabilities() const {
        return torch::sigmoid(log_alpha_);
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "ConcreteDropout(probability_shape=" << log_alpha_.sizes()
               << ", temperature=" << temperature_
               << ", dropout_regularizer_factor=" << dropout_regularizer_factor_;
        if (log_alpha_.numel() > 0) {
             stream << ", current_p_mean_approx=" << torch::sigmoid(log_alpha_.mean()).item<double>();
        }
        stream << ")";
    }
};

TORCH_MODULE(ConcreteDropout); // Creates the ConcreteDropout module "class"

/*
// Example of how to use the ConcreteDropout module:
// (This is for illustration and would typically be in your main application code)

#include <iostream>
#include <torch/optim.hpp> // For torch::optim::SGD or Adam

void run_concrete_dropout_example() {
    torch::manual_seed(0); // For reproducible results

    // --- Configuration ---
    double initial_p_value = 0.2;       // Target initial dropout rate
    double temperature_value = 0.1;     // Temperature for Concrete distribution
    double reg_factor_value = 1e-4;   // Factor for the dropout regularization term

    // 1. ConcreteDropout with a scalar (global) learnable dropout rate
    ConcreteDropout global_concrete_dropout_module(
        {}, // Empty shape for scalar log_alpha
        initial_p_value,
        temperature_value,
        reg_factor_value
    );
    std::cout << "Global ConcreteDropout Module: " << global_concrete_dropout_module << std::endl;
    std::cout << "Initial log_alpha: " << global_concrete_dropout_module->log_alpha_ << std::endl;
    std::cout << "Initial p (dropout probability): " << global_concrete_dropout_module->get_dropout_probabilities() << std::endl;


    torch::Tensor input_tensor = torch::randn({3, 5}); // Example input: Batch=3, Features=5
    global_concrete_dropout_module->train(); // Set to training mode for dropout to be active

    // --- Simulate a training step to see how log_alpha might be updated ---
    // In a real application, this module would be part of a larger neural network.
    torch::optim::Adam optimizer(
        global_concrete_dropout_module->parameters(), // Pass parameters of this module to optimizer
        torch::optim::AdamOptions(1e-2) // Learning rate
    );

    optimizer.zero_grad(); // Clear previous gradients

    // Forward pass through the ConcreteDropout module
    torch::Tensor output_during_train = global_concrete_dropout_module->forward(input_tensor);

    // Assume a dummy main loss from a downstream task (e.g., we want the output's mean to be high)
    torch::Tensor main_task_loss = -output_during_train.mean(); // Negative mean to maximize it

    // Calculate and add the Concrete Dropout regularization term
    torch::Tensor concrete_regularization_loss = global_concrete_dropout_module->calculate_regularization_term();
    torch::Tensor total_loss = main_task_loss + concrete_regularization_loss;

    total_loss.backward(); // Compute gradients
    optimizer.step();      // Update log_alpha

    std::cout << "\n--- After one optimization step ---" << std::endl;
    std::cout << "Input Tensor:\n" << input_tensor << std::endl;
    std::cout << "Output (during training, before opt step):\n" << output_during_train << std::endl;
    std::cout << "Main Task Loss: " << main_task_loss.item<double>() << std::endl;
    std::cout << "Concrete Regularization Loss: " << concrete_regularization_loss.item<double>() << std::endl;
    std::cout << "Total Loss: " << total_loss.item<double>() << std::endl;
    std::cout << "log_alpha after optimization step: " << global_concrete_dropout_module->log_alpha_ << std::endl;
    std::cout << "p (dropout probability) after optimization step: "
              << global_concrete_dropout_module->get_dropout_probabilities() << std::endl;


    // --- Evaluation mode ---
    global_concrete_dropout_module->eval(); // Set to evaluation mode
    torch::Tensor output_during_eval = global_concrete_dropout_module->forward(input_tensor);
    std::cout << "\nOutput (evaluation mode):\n" << output_during_eval << std::endl;
    // Expected: output_during_eval should be identical to input_tensor in eval mode.


    // 2. ConcreteDropout with per-feature learnable dropout rates
    int num_features = 4;
    ConcreteDropout per_feature_dropout_module(
        {num_features}, // p_shape for per-feature: [num_features]
        0.1,            // Initial dropout rate for all features
        temperature_value,
        reg_factor_value
    );
    std::cout << "\nPer-Feature ConcreteDropout Module: " << per_feature_dropout_module << std::endl;
    std::cout << "Initial log_alphas (per-feature): " << per_feature_dropout_module->log_alpha_ << std::endl;
    std::cout << "Initial ps (per-feature): " << per_feature_dropout_module->get_dropout_probabilities() << std::endl;

    torch::Tensor feature_example_input = torch::randn({2, num_features}); // Batch=2, Features=4
    per_feature_dropout_module->train();
    torch::Tensor feature_output_train = per_feature_dropout_module->forward(feature_example_input);
    torch::Tensor feature_reg_loss = per_feature_dropout_module->calculate_regularization_term();

    std::cout << "Feature Example Input (Batch=2, Features=4):\n" << feature_example_input << std::endl;
    std::cout << "Feature Output (train, per-feature p):\n" << feature_output_train << std::endl;
    std::cout << "Feature-wise Regularization Loss: " << feature_reg_loss.item<double>() << std::endl;
}

// int main() {
//    run_concrete_dropout_example();
//    return 0;
// }
*/




namespace xt::dropouts
{
    torch::Tensor concrete_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto ConcreteDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::concrete_dropout(torch::zeros(10));
    }
}
