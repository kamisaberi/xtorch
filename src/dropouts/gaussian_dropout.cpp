#include "include/dropouts/gaussian_dropout.h"


//
// #include <torch/torch.h>
// #include <cmath>   // For std::sqrt
// #include <ostream> // For std::ostream
//
// struct GaussianDropoutImpl : torch::nn::Module {
//     double p_rate_; // Corresponds to the 'p' in standard dropout, used to calculate alpha (variance)
//     double alpha_;  // Variance of the multiplicative noise (noise ~ N(0, alpha_)), so multiplier ~ N(1, alpha_)
//
//     GaussianDropoutImpl(double p_rate = 0.1) : p_rate_(p_rate) {
//         TORCH_CHECK(p_rate_ >= 0.0 && p_rate_ < 1.0,
//                     "GaussianDropout p_rate must be between 0 and 1 (exclusive of 1).");
//         if (p_rate_ == 0.0) {
//             alpha_ = 0.0;
//         } else {
//             // alpha = p / (1-p) as suggested in the original dropout paper
//             alpha_ = p_rate_ / (1.0 - p_rate_);
//         }
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || p_rate_ == 0.0) {
//             // If not in training mode or if effective dropout rate is zero,
//             // return the input as is.
//             return input;
//         }
//
//         // Generate multiplicative noise.
//         // We want noise ~ N(1, alpha).
//         // This can be achieved by generating noise_std_normal ~ N(0, 1),
//         // then scaling: 1 + noise_std_normal * sqrt(alpha).
//
//         // randn_like generates noise from N(0, 1)
//         torch::Tensor noise_std_normal = torch::randn_like(input);
//
//         // Multiplicative noise: 1 + N(0, alpha) = N(1, alpha)
//         torch::Tensor multiplicative_noise = 1.0 + (noise_std_normal * std::sqrt(alpha_));
//
//         return input * multiplicative_noise;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "GaussianDropout(p_rate=" << p_rate_ << ", alpha=" << alpha_ << ")";
//     }
// };
//
// TORCH_MODULE(GaussianDropout); // Creates the GaussianDropout module "class"
//
// /*
// // Example of how to use the GaussianDropout module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
//
// void run_gaussian_dropout_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     double dropout_param_p = 0.25; // This 'p' leads to alpha = 0.25 / (1-0.25) = 0.25 / 0.75 = 1/3
//     GaussianDropout dropout_module(dropout_param_p);
//     std::cout << "GaussianDropout Module: " << dropout_module << std::endl;
//     // Expected alpha approx 0.333
//
//     // Example input tensor
//     torch::Tensor input_tensor = torch::ones({2, 5}); // Batch_size=2, Features=5
//     std::cout << "Input Tensor (all ones):\n" << input_tensor << std::endl;
//
//     // --- Training mode ---
//     dropout_module->train(); // Set the module to training mode
//     torch::Tensor output_train = dropout_module->forward(input_tensor);
//     std::cout << "Output (training mode, p_rate=" << dropout_param_p << "):\n" << output_train << std::endl;
//     // Expected: Output elements will be the input elements (1.0) multiplied by
//     // random numbers drawn from N(1, alpha).
//     // The mean of the output elements should be around 1.0.
//     std::cout << "Mean of output_train: " << output_train.mean().item<double>() << std::endl;
//
//
//     // --- Evaluation mode ---
//     dropout_module->eval(); // Set the module to evaluation mode
//     torch::Tensor output_eval = dropout_module->forward(input_tensor);
//     std::cout << "Output (evaluation mode):\n" << output_eval << std::endl;
//     // Expected: Output should be identical to the input tensor in evaluation mode.
//     TORCH_CHECK(torch::allclose(input_tensor, output_eval), "GaussianDropout eval output mismatch!");
//
//
//     // --- Test with p_rate = 0.0 (no dropout) ---
//     GaussianDropout no_dropout_module(0.0);
//     std::cout << "\nGaussianDropout Module (p_rate=0.0): " << no_dropout_module << std::endl; // alpha should be 0
//     no_dropout_module->train();
//     torch::Tensor output_no_drop = no_dropout_module->forward(input_tensor);
//     std::cout << "Output (training mode, p_rate=0.0):\n" << output_no_drop << std::endl;
//     TORCH_CHECK(torch::allclose(input_tensor, output_no_drop), "GaussianDropout p_rate=0.0 output mismatch!");
//
//
//     // --- Test with a slightly higher p_rate to see more variance ---
//     GaussianDropout high_var_dropout_module(0.5); // alpha = 0.5 / 0.5 = 1.0
//     std::cout << "\nGaussianDropout Module (p_rate=0.5): " << high_var_dropout_module << std::endl; // alpha should be 1.0
//     high_var_dropout_module->train();
//     torch::Tensor output_high_var = high_var_dropout_module->forward(input_tensor);
//     std::cout << "Output (training mode, p_rate=0.5):\n" << output_high_var << std::endl;
//     std::cout << "Mean of output_high_var: " << output_high_var.mean().item<double>() << std::endl;
//     std::cout << "Std of output_high_var: " << output_high_var.std().item<double>() << std::endl;
//     // With input=1 and alpha=1, output elements are ~N(1,1).
//     // The standard deviation of the output should be around sqrt(alpha) * input_value = sqrt(1)*1 = 1.
//
// }
//
// // int main() {
// //    run_gaussian_dropout_example();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    GaussianDropout::GaussianDropout(double p_rate ) : p_rate_(p_rate)
    {
        TORCH_CHECK(p_rate_ >= 0.0 && p_rate_ < 1.0,
                    "GaussianDropout p_rate must be between 0 and 1 (exclusive of 1).");
        if (p_rate_ == 0.0)
        {
            alpha_ = 0.0;
        }
        else
        {
            // alpha = p / (1-p) as suggested in the original dropout paper
            alpha_ = p_rate_ / (1.0 - p_rate_);
        }
    }


    auto GaussianDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training() || p_rate_ == 0.0)
        {
            // If not in training mode or if effective dropout rate is zero,
            // return the input as is.
            return input;
        }

        // Generate multiplicative noise.
        // We want noise ~ N(1, alpha).
        // This can be achieved by generating noise_std_normal ~ N(0, 1),
        // then scaling: 1 + noise_std_normal * sqrt(alpha).

        // randn_like generates noise from N(0, 1)
        torch::Tensor noise_std_normal = torch::randn_like(input);

        // Multiplicative noise: 1 + N(0, alpha) = N(1, alpha)
        torch::Tensor multiplicative_noise = 1.0 + (noise_std_normal * std::sqrt(alpha_));

        return input * multiplicative_noise;



    }
}
