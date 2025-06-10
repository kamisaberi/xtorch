#include "include/dropouts/auto_dropout.h"


// #include <torch/torch.h>
// #include <vector>    // For std::vector
// #include <cmath>     // For std::log
// #include <ostream>   // For std::ostream
//
// namespace { // Anonymous namespace for helper utility
// // Helper to calculate initial log_alpha from an initial dropout rate p.
// // p = sigmoid(log_alpha)  => log_alpha = log(p / (1-p))
// double calculate_initial_log_alpha_value(double initial_dropout_rate) {
//     // Clamp initial_dropout_rate to avoid log(0) or log( division by zero )
//     double epsilon = 1e-7; // A small epsilon
//     if (initial_dropout_rate < epsilon) {
//         initial_dropout_rate = epsilon;
//     }
//     if (initial_dropout_rate > 1.0 - epsilon) {
//         initial_dropout_rate = 1.0 - epsilon;
//     }
//     return std::log(initial_dropout_rate / (1.0 - initial_dropout_rate));
// }
// } // namespace
//
//
// struct AutoDropoutImpl : torch::nn::Module {
//     // Learnable parameter(s) from which dropout probability is derived.
//     // log_alpha is unconstrained, p = sigmoid(log_alpha) is in (0, 1).
//     torch::Tensor log_alpha_;
//
//     // probability_shape: Determines if log_alpha is scalar (global dropout rate)
//     // or a tensor (e.g., per-feature, per-channel dropout rates).
//     // initial_dropout_rate: The desired initial dropout probability p.
//     AutoDropoutImpl(c10::IntArrayRef probability_shape = {}, double initial_dropout_rate = 0.05) {
//         double initial_log_alpha_val = calculate_initial_log_alpha_value(initial_dropout_rate);
//
//         torch::Tensor log_alpha_init;
//         if (probability_shape.empty()) { // Scalar probability
//             log_alpha_init = torch::tensor(initial_log_alpha_val, torch::kFloat32);
//         } else {
//             // Create a tensor of the given shape, filled with the initial log_alpha value.
//             log_alpha_init = torch::full(probability_shape, initial_log_alpha_val, torch::kFloat32);
//         }
//         log_alpha_ = register_parameter("log_alpha", log_alpha_init);
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training()) {
//             return input; // During evaluation, dropout is inactive.
//         }
//
//         // Calculate dropout probability p from log_alpha.
//         // p will have gradients flowing back to log_alpha_ due to sigmoid.
//         torch::Tensor p = torch::sigmoid(log_alpha_);
//
//         // Clamp p to prevent keep_prob from being too close to 0 or 1, ensuring numerical stability.
//         // Sigmoid output is (0,1), so p is already >0 and <1 if log_alpha_ is finite.
//         // This clamp is a safeguard or for when very extreme log_alpha values might occur.
//         // Crucially, it ensures (1-p) is not zero for the division.
//         p = torch::clamp(p, 0.0, 1.0 - 1e-7); // Ensure p < 1.0
//
//         torch::Tensor keep_prob = 1.0 - p; // Probability of NOT dropping out.
//         torch::Tensor kp_broadcastable = keep_prob;
//
//         // Heuristic for broadcasting keep_prob (e.g., for per-channel dropout in conv layers).
//         // If keep_prob is 1D (e.g. shape [C]) and input is multi-dimensional (e.g. [N, C, H, W]),
//         // and its size matches input's dim 1 (assumed channel dimension), reshape for broadcasting.
//         if (keep_prob.dim() == 1 && input.dim() > 1 && keep_prob.size(0) == input.size(1)) {
//             std::vector<int64_t> view_shape(input.dim(), 1L); // Create shape like [1, 1, ..., 1]
//             view_shape[1] = keep_prob.size(0); // Set channel dimension, e.g., [1, C, 1, 1]
//             kp_broadcastable = keep_prob.view(view_shape);
//         }
//         // If shapes are otherwise, rely on standard PyTorch broadcasting rules.
//         // If incompatible shapes that aren't handled by the heuristic, PyTorch will raise an error.
//
//         // Standard inverted dropout:
//         // 1. Generate a random tensor with the same shape as input.
//         // 2. Create a binary mask: 1 if random_val < keep_prob, 0 otherwise.
//         // 3. Apply mask.
//         // 4. Scale by 1/keep_prob.
//         torch::Tensor random_tensor = torch::rand_like(input);
//         torch::Tensor mask = (random_tensor < kp_broadcastable).to(input.dtype());
//
//         // kp_broadcastable is guaranteed to be >= 1e-7 due to p clamping.
//         torch::Tensor output = (input * mask) / kp_broadcastable;
//
//         return output;
//     }
//
//     // Optional: A method to get the current dropout probability p(s).
//     // This could be useful for monitoring or for calculating regularization terms
//     // as in Concrete Dropout (which would be done outside this module, in the training loop).
//     torch::Tensor get_dropout_probabilities() const {
//         return torch::sigmoid(log_alpha_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "AutoDropout(probability_shape=" << log_alpha_.sizes()
//                << ", current_p_approx=" << torch::sigmoid(log_alpha_.mean()).item<double>() // Show average p
//                << ")";
//     }
// };
//
// TORCH_MODULE(AutoDropout); // Creates the AutoDropout module "class"
//
// /*
// // Example of how to use the AutoDropout module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
// #include <torch/optim. Apesar> // For torch::optim::SGD
//
// void run_auto_dropout_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     // 1. AutoDropout with a scalar (global) learnable dropout rate
//     AutoDropout global_dropout_module( {}, 0.5 ); // Empty shape for scalar, initial p approx 0.5
//     std::cout << "Global AutoDropout Module: " << global_dropout_module << std::endl;
//     std::cout << "Initial log_alpha: " << global_dropout_module->log_alpha_ << std::endl;
//     std::cout << "Initial p: " << global_dropout_module->get_dropout_probabilities() << std::endl;
//
//
//     torch::Tensor input_tensor = torch::ones({2, 4}); // Example input
//     global_dropout_module->train(); // Set to training mode
//     torch::Tensor output_train = global_dropout_module->forward(input_tensor);
//     std::cout << "Input:\n" << input_tensor << std::endl;
//     std::cout << "Output (train, global p):\n" << output_train << std::endl;
//
//     // To actually learn log_alpha, you'd need a loss and an optimizer
//     // For example:
//     // torch::optim::SGD optimizer(global_dropout_module->parameters(), 0.1);
//     // torch::Tensor loss = output_train.sum(); // Dummy loss
//     // optimizer.zero_grad();
//     // loss.backward();
//     // optimizer.step();
//     // std::cout << "log_alpha after one step: " << global_dropout_module->log_alpha_ << std::endl;
//     // std::cout << "p after one step: " << global_dropout_module->get_dropout_probabilities() << std::endl;
//
//
//     global_dropout_module->eval(); // Set to evaluation mode
//     torch::Tensor output_eval = global_dropout_module->forward(input_tensor);
//     std::cout << "Output (eval, global p):\n" << output_eval << std::endl; // Should be same as input
//
//
//     // 2. AutoDropout with per-feature learnable dropout rates
//     // For an input of shape (Batch, Features), let p_shape be (Features)
//     int num_features = 3;
//     AutoDropout per_feature_dropout_module( {num_features}, 0.1 ); // p_shape {3}, initial p approx 0.1
//     std::cout << "\nPer-Feature AutoDropout Module: " << per_feature_dropout_module << std::endl;
//     std::cout << "Initial log_alphas (per-feature): " << per_feature_dropout_module->log_alpha_ << std::endl;
//     std::cout << "Initial ps (per-feature): " << per_feature_dropout_module->get_dropout_probabilities() << std::endl;
//
//
//     torch::Tensor feature_input = torch::randn({5, num_features}); // Batch=5, Features=3
//     per_feature_dropout_module->train();
//     torch::Tensor feature_output_train = per_feature_dropout_module->forward(feature_input);
//     std::cout << "Feature Input:\n" << feature_input << std::endl;
//     std::cout << "Feature Output (train, per-feature p):\n" << feature_output_train << std::endl;
//
//
//     // 3. AutoDropout for convolutional layers (per-channel learnable dropout rates)
//     // For an input of shape (N, C, H, W), let p_shape be (C)
//     int num_channels = 3;
//     AutoDropout per_channel_dropout_module( {num_channels}, 0.2 ); // p_shape {3} -> [C], initial p approx 0.2
//     std::cout << "\nPer-Channel AutoDropout Module: " << per_channel_dropout_module << std::endl;
//     std::cout << "Initial log_alphas (per-channel): " << per_channel_dropout_module->log_alpha_ << std::endl;
//
//     torch::Tensor conv_input = torch::randn({2, num_channels, 5, 5}); // N=2, C=3, H=5, W=5
//     per_channel_dropout_module->train();
//     torch::Tensor conv_output_train = per_channel_dropout_module->forward(conv_input);
//     std::cout << "Conv Input shape: " << conv_input.sizes() << std::endl;
//     std::cout << "Conv Output (train, per-channel p) shape: " << conv_output_train.sizes() << std::endl;
//     // The broadcasting heuristic should make keep_prob [1, C, 1, 1]
//     std::cout << "Current per-channel p values: " << per_channel_dropout_module->get_dropout_probabilities() << std::endl;
//
// }
//
// // int main() {
// //    run_auto_dropout_example();
// //    return 0;
// // }
// */



namespace xt::dropouts
{
    torch::Tensor auto_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto AutoDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::auto_dropout(torch::zeros(10));
    }
}
