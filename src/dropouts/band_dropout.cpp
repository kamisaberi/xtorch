#include "include/dropouts/band_dropout.h"


// #include <torch/torch.h>
// #include <vector>
// #include <cmath>     // For std::log
// #include <ostream>   // For std::ostream
//
// namespace { // Anonymous namespace for helper utility
// double calculate_logit_from_probability(double probability, double epsilon = 1e-7) {
//     if (probability < epsilon) probability = epsilon;
//     if (probability > 1.0 - epsilon) probability = 1.0 - epsilon;
//     return std::log(probability / (1.0 - probability));
// }
// } // namespace
//
// struct BendDropoutImpl : torch::nn::Module {
//     torch::Tensor alpha_;
//     torch::Tensor beta_;
//     double epsilon_ = 1e-7; // For numerical stability
//
//     BendDropoutImpl(c10::IntArrayRef params_shape = {},
//                     double initial_baseline_dropout_rate = 0.1,
//                     double initial_alpha_value = 0.01) {
//
//         double initial_keep_prob_baseline = 1.0 - initial_baseline_dropout_rate;
//         double initial_beta_val = calculate_logit_from_probability(initial_keep_prob_baseline, epsilon_);
//
//         torch::Tensor alpha_init;
//         torch::Tensor beta_init;
//
//         if (params_shape.empty()) {
//             alpha_init = torch::tensor(initial_alpha_value, torch::kFloat32);
//             beta_init = torch::tensor(initial_beta_val, torch::kFloat32);
//         } else {
//             alpha_init = torch::full(params_shape, initial_alpha_value, torch::kFloat32);
//             beta_init = torch::full(params_shape, initial_beta_val, torch::kFloat32);
//         }
//         alpha_ = register_parameter("alpha", alpha_init);
//         beta_ = register_parameter("beta", beta_init);
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training()) {
//             return input;
//         }
//
//         torch::Tensor current_alpha = alpha_;
//         torch::Tensor current_beta = beta_;
//
//         if (current_alpha.dim() == 1 && input.dim() > 1 && current_alpha.size(0) == input.size(1) && input.size(1) > 0) {
//             std::vector<int64_t> view_shape(input.dim(), 1L);
//             view_shape[1] = current_alpha.size(0);
//             current_alpha = current_alpha.view(view_shape);
//             if (current_beta.dim() == 1 && current_beta.size(0) == input.size(1)) {
//                  current_beta = current_beta.view(view_shape);
//             }
//         }
//
//         torch::Tensor keep_prob_logits = current_alpha * torch::abs(input) + current_beta;
//         torch::Tensor keep_probabilities = torch::sigmoid(keep_prob_logits);
//         keep_probabilities = torch::clamp(keep_probabilities, epsilon_, 1.0);
//
//         torch::Tensor random_tensor = torch::rand_like(input);
//         torch::Tensor mask = (random_tensor < keep_probabilities).to(input.dtype());
//
//         torch::Tensor output = (input * mask) / keep_probabilities;
//
//         return output;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "BendDropout(params_shape=" << alpha_.sizes();
//         if (alpha_.numel() > 0 && alpha_.numel() == beta_.numel() && alpha_.is_ सेम_size(beta_)) {
//              if (alpha_.numel() == 1) {
//                  stream << ", alpha=" << alpha_.item<double>()
//                         << ", beta=" << beta_.item<double>();
//              } else {
//                  stream << ", alpha_mean=" << alpha_.mean().item<double>()
//                         << ", beta_mean=" << beta_.mean().item<double>();
//              }
//         }
//         stream << ")";
//     }
// };
//
// TORCH_MODULE(BendDropout);
//
// /*
// // Example of how to use the BendDropout module:
// #include <iostream>
//
// void run_bend_dropout_example() {
//     torch::manual_seed(0);
//
//     // 1. BendDropout with scalar parameters
//     BendDropout scalar_bend_dropout_module({}, 0.2, 0.5); // baseline_drop_rate=0.2, alpha=0.5
//     std::cout << "Scalar BendDropout Module: " << scalar_bend_dropout_module << std::endl;
//
//     torch::Tensor input1 = torch::tensor({{-2.0, -0.1, 0.0, 0.1, 2.0}}, torch::kFloat32);
//     scalar_bend_dropout_module->train();
//     torch::Tensor output1_train = scalar_bend_dropout_module->forward(input1);
//     std::cout << "Input 1:\n" << input1 << std::endl;
//     // For input1, abs(input) varies.
//     // keep_prob = sigmoid(0.5 * abs(input) + logit(1-0.2))
//     // logit(0.8) approx 1.386
//     // For input -2.0: sigmoid(0.5*2 + 1.386) = sigmoid(2.386) approx 0.915 (high keep_prob)
//     // For input -0.1: sigmoid(0.5*0.1 + 1.386) = sigmoid(0.05 + 1.386) = sigmoid(1.436) approx 0.808
//     // For input  0.0: sigmoid(1.386) approx 0.8 (baseline keep_prob)
//     std::cout << "Output 1 (train):\n" << output1_train << std::endl;
//
//     scalar_bend_dropout_module->eval();
//     torch::Tensor output1_eval = scalar_bend_dropout_module->forward(input1);
//     std::cout << "Output 1 (eval):\n" << output1_eval << std::endl;
//
//
//     // 2. BendDropout with per-channel parameters for a 2D-like input (Batch, Features)
//     int num_features = 3;
//     BendDropout per_feature_bend_dropout_module({num_features}, 0.1, 0.2);
//     std::cout << "\nPer-Feature BendDropout Module: " << per_feature_bend_dropout_module << std::endl;
//     // Manually set different alpha/beta for illustration if desired
//     // per_feature_bend_dropout_module->alpha.data() = torch::tensor({0.1, 0.5, 1.0});
//     // per_feature_bend_dropout_module->beta.data()  = torch::tensor({logit(0.9), logit(0.8), logit(0.7)});
//
//
//     torch::Tensor input2 = torch::randn({2, num_features}); // Batch=2, Features=3
//     per_feature_bend_dropout_module->train();
//     torch::Tensor output2_train = per_feature_bend_dropout_module->forward(input2);
//     std::cout << "Input 2:\n" << input2 << std::endl;
//     std::cout << "Output 2 (train, per-feature):\n" << output2_train << std::endl;
//
//     // 3. BendDropout for Conv-like input (N, C, H, W) with per-channel parameters
//     int num_channels = 2;
//     BendDropout per_channel_bend_dropout_module({num_channels}, 0.25, 0.3);
//     std::cout << "\nPer-Channel BendDropout Module: " << per_channel_bend_dropout_module << std::endl;
//
//     torch::Tensor input3 = torch::randn({1, num_channels, 2, 2}); // N=1, C=2, H=2, W=2
//     per_channel_bend_dropout_module->train();
//     torch::Tensor output3_train = per_channel_bend_dropout_module->forward(input3);
//     std::cout << "Input 3 (NCHW):\n" << input3 << std::endl;
//     std::cout << "Output 3 (train, per-channel NCHW):\n" << output3_train << std::endl;
// }
//
// // int main() {
// //     run_bend_dropout_example();
// //     return 0;
// // }
// */


namespace xt::dropouts
{
    torch::Tensor band_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto BandDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::band_dropout(torch::zeros(10));
    }
}
