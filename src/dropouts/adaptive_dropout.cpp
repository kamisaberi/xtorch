#include <dropouts/adaptive_dropout.h>


// #include <torch/torch.h>
// #include <vector>
// #include <cmath>     // For std::log
// #include <ostream>   // For std::ostream
//
// namespace { // Anonymous namespace for helper utility
// double calculate_initial_log_alpha_value(double initial_dropout_rate) {
//     // Clamp initial_dropout_rate to avoid log(0) or log( división by zero )
//     double epsilon = 1e-7;
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
// struct AdaptiveDropoutImpl : torch::nn::Module {
//     torch::Tensor log_alpha_;
//
//     AdaptiveDropoutImpl(c10::IntArrayRef probability_shape = {}, double initial_dropout_rate = 0.05) {
//         double initial_log_alpha_val = calculate_initial_log_alpha_value(initial_dropout_rate);
//
//         torch::Tensor log_alpha_init;
//         if (probability_shape.empty()) { // Scalar probability
//             log_alpha_init = torch::tensor(initial_log_alpha_val, torch::kFloat32);
//         } else {
//             log_alpha_init = torch::full(probability_shape, initial_log_alpha_val, torch::kFloat32);
//         }
//         log_alpha_ = register_parameter("log_alpha", log_alpha_init);
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training()) {
//             return input;
//         }
//
//         torch::Tensor p = torch::sigmoid(log_alpha_);
//
//         // Clamp p to prevent keep_prob from being too close to 0 or 1, ensuring numerical stability.
//         // sigmoid output is (0,1), so p is already >0 and <1 if log_alpha_ is finite.
//         // Clamping mainly for extreme log_alpha_ values or if desired for stricter bounds.
//         // Let's ensure keep_prob is not zero for division.
//         p = torch::clamp(p, 0.0, 1.0 - 1e-7);
//
//         torch::Tensor keep_prob = 1.0 - p;
//         torch::Tensor kp_broadcastable = keep_prob;
//
//         // Heuristic: if keep_prob is 1D (e.g. shape [C]) and input is multi-dimensional (e.g. [N, C, H, W]),
//         // and its size matches input's dim 1 (channel dimension), reshape for broadcasting.
//         if (keep_prob.dim() == 1 && input.dim() > 1 && keep_prob.size(0) == input.size(1)) {
//             std::vector<int64_t> view_shape(input.dim(), 1L); // Create shape like [1, 1, ..., 1]
//             view_shape[1] = keep_prob.size(0); // Set channel dimension, e.g., [1, C, 1, 1]
//             kp_broadcastable = keep_prob.view(view_shape);
//         }
//         // If shapes are otherwise, rely on standard PyTorch broadcasting rules.
//         // If incompatible, PyTorch will raise an error.
//
//         torch::Tensor random_tensor = torch::rand_like(input);
//         torch::Tensor mask = (random_tensor < kp_broadcastable).to(input.dtype());
//
//         // Scale the output by 1/keep_prob.
//         // kp_broadcastable is guaranteed to be >= 1e-7 due to p clamping.
//         torch::Tensor output = (input * mask) / kp_broadcastable;
//
//         return output;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "AdaptiveDropout(probability_shape=" << log_alpha_.sizes() << ")";
//     }
// };
//
// TORCH_MODULE(AdaptiveDropout); // Creates the AdaptiveDropout module "class"
//
// /*
// // Example of how to use the AdaptiveDropout module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
//
// void run_adaptive_dropout_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     // 1. AdaptiveDropout with a scalar (global) learnable dropout rate
//     AdaptiveDropout global_dropout_module( {}, 0.5 ); // Empty shape for scalar, initial p=0.5
//     std::cout << "Global Dropout Module: " << global_dropout_module << std::endl;
//
//     torch::Tensor input_tensor = torch::ones({2, 4}); // Example input
//     global_dropout_module->train(); // Set to training mode
//     torch::Tensor output_train = global_dropout_module->forward(input_tensor);
//     std::cout << "Input:\n" << input_tensor << std::endl;
//     std::cout << "Output (train, global p):\n" << output_train << std::endl;
//     // To access and potentially see gradients (after a backward pass):
//     // std::cout << "Initial log_alpha: " << global_dropout_module->log_alpha_ << std::endl;
//
//
//     global_dropout_module->eval(); // Set to evaluation mode
//     torch::Tensor output_eval = global_dropout_module->forward(input_tensor);
//     std::cout << "Output (eval, global p):\n" << output_eval << std::endl; // Should be same as input
//
//
//     // 2. AdaptiveDropout with per-feature learnable dropout rates
//     // For an input of shape (Batch, Features), let p_shape be (Features)
//     int num_features = 3;
//     AdaptiveDropout per_feature_dropout_module( {num_features}, 0.1 ); // p_shape {3}, initial p=0.1
//     std::cout << "\nPer-Feature Dropout Module: " << per_feature_dropout_module << std::endl;
//
//     torch::Tensor feature_input = torch::randn({2, num_features});
//     per_feature_dropout_module->train();
//     torch::Tensor feature_output_train = per_feature_dropout_module->forward(feature_input);
//     std::cout << "Feature Input:\n" << feature_input << std::endl;
//     std::cout << "Feature Output (train, per-feature p):\n" << feature_output_train << std::endl;
//
//
//     // 3. AdaptiveDropout for convolutional layers (per-channel dropout rates)
//     // For an input of shape (N, C, H, W), let p_shape be (C)
//     int num_channels = 3;
//     AdaptiveDropout per_channel_dropout_module( {num_channels}, 0.2 ); // p_shape {3}, initial p=0.2
//     std::cout << "\nPer-Channel Dropout Module: " << per_channel_dropout_module << std::endl;
//
//     torch::Tensor conv_input = torch::randn({1, num_channels, 2, 2}); // N=1, C=3, H=2, W=2
//     per_channel_dropout_module->train();
//     torch::Tensor conv_output_train = per_channel_dropout_module->forward(conv_input);
//     std::cout << "Conv Input:\n" << conv_input << std::endl;
//     std::cout << "Conv Output (train, per-channel p):\n" << conv_output_train << std::endl;
//     // The heuristic for keep_prob.dim()==1 will reshape p from [C] to [1,C,1,1] for broadcasting.
// }
//
// // int main() {
// //     run_adaptive_dropout_example();
// //     return 0;
// // }
// */


namespace xt::dropouts
{
    namespace
    {
        // Anonymous namespace for helper utility
        double calculate_initial_log_alpha_value(double initial_dropout_rate)
        {
            // Clamp initial_dropout_rate to avoid log(0) or log( división by zero )
            double epsilon = 1e-7;
            if (initial_dropout_rate < epsilon)
            {
                initial_dropout_rate = epsilon;
            }
            if (initial_dropout_rate > 1.0 - epsilon)
            {
                initial_dropout_rate = 1.0 - epsilon;
            }
            return std::log(initial_dropout_rate / (1.0 - initial_dropout_rate));
        }
    }


    AdaptiveDropout::AdaptiveDropout(c10::IntArrayRef probability_shape, double initial_dropout_rate)
    {
        double initial_log_alpha_val = calculate_initial_log_alpha_value(initial_dropout_rate);

        torch::Tensor log_alpha_init;
        if (probability_shape.empty())
        {
            // Scalar probability
            log_alpha_init = torch::tensor(initial_log_alpha_val, torch::kFloat32);
        }
        else
        {
            log_alpha_init = torch::full(probability_shape, initial_log_alpha_val, torch::kFloat32);
        }
        log_alpha_ = register_parameter("log_alpha", log_alpha_init);
    }

    auto AdaptiveDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training())
        {
            return input;
        }

        torch::Tensor p = torch::sigmoid(log_alpha_);

        // Clamp p to prevent keep_prob from being too close to 0 or 1, ensuring numerical stability.
        // sigmoid output is (0,1), so p is already >0 and <1 if log_alpha_ is finite.
        // Clamping mainly for extreme log_alpha_ values or if desired for stricter bounds.
        // Let's ensure keep_prob is not zero for division.
        p = torch::clamp(p, 0.0, 1.0 - 1e-7);

        torch::Tensor keep_prob = 1.0 - p;
        torch::Tensor kp_broadcastable = keep_prob;

        // Heuristic: if keep_prob is 1D (e.g. shape [C]) and input is multi-dimensional (e.g. [N, C, H, W]),
        // and its size matches input's dim 1 (channel dimension), reshape for broadcasting.
        if (keep_prob.dim() == 1 && input.dim() > 1 && keep_prob.size(0) == input.size(1))
        {
            std::vector<int64_t> view_shape(input.dim(), 1L); // Create shape like [1, 1, ..., 1]
            view_shape[1] = keep_prob.size(0); // Set channel dimension, e.g., [1, C, 1, 1]
            kp_broadcastable = keep_prob.view(view_shape);
        }
        // If shapes are otherwise, rely on standard PyTorch broadcasting rules.
        // If incompatible, PyTorch will raise an error.

        torch::Tensor random_tensor = torch::rand_like(input);
        torch::Tensor mask = (random_tensor < kp_broadcastable).to(input.dtype());

        // Scale the output by 1/keep_prob.
        // kp_broadcastable is guaranteed to be >= 1e-7 due to p clamping.
        torch::Tensor output = (input * mask) / kp_broadcastable;

        return output;

        return xt::dropouts::adaptive_dropout(torch::zeros(10));
    }
}
