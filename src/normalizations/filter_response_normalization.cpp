#include "include/normalizations/filter_response_normalization.h"


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct FilterResponseNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct FilterResponseNormalization : torch::nn::ModuleHolder<FilterResponseNormalizationImpl> {
//     using torch::nn::ModuleHolder<FilterResponseNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for FilterResponseNormalization + TLU
// struct FilterResponseNormalizationImpl : torch::nn::Module {
//     int64_t num_features_; // Number of channels C
//     double eps_;           // Small epsilon for numerical stability in FRN
//
//     // Learnable parameters for FRN (affine transformation)
//     torch::Tensor gamma_;  // Per-channel scale
//     torch::Tensor beta_;   // Per-channel shift
//
//     // Learnable parameter for TLU (Thresholded Linear Unit)
//     torch::Tensor tau_;    // Per-channel threshold
//
//     FilterResponseNormalizationImpl(int64_t num_features, double eps = 1e-6)
//         : num_features_(num_features),
//           eps_(eps) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//
//         // Initialize learnable parameters
//         // gamma and beta are similar to BatchNorm's affine parameters
//         gamma_ = register_parameter("gamma", torch::ones({1, num_features_, 1, 1})); // Shape for broadcasting with NCHW
//         beta_  = register_parameter("beta",  torch::zeros({1, num_features_, 1, 1})); // Shape for broadcasting
//
//         // tau is the learnable threshold for TLU
//         // Initializing tau to zero means TLU initially behaves like ReLU around zero if gamma > 0
//         tau_   = register_parameter("tau",   torch::zeros({1, num_features_, 1, 1})); // Shape for broadcasting
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x is expected to be 4D: (N, C, H, W)
//         // C must be num_features_
//         TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
//
//         // --- 1. FRN: Normalization Step ---
//         // Calculate sum of squares over spatial dimensions (H, W) for each channel and each batch instance.
//         // x_squared = x^2
//         torch::Tensor x_squared = x.pow(2);
//
//         // Sum over spatial dimensions (H, W), which are dims 2 and 3.
//         // keepdim=true to maintain shape for broadcasting.
//         // nu2 = sum_{h,w} x^2_{n,c,h,w} / (H*W)  (mean square)
//         torch::Tensor nu2 = x_squared.mean(std::vector<int64_t>{2, 3}, /*keepdim=*/true);
//         // nu2 shape: (N, C, 1, 1)
//
//         // Normalize x: x_hat = x / sqrt(nu2 + eps)
//         torch::Tensor x_frn = x * torch::rsqrt(nu2 + eps_); // Using rsqrt for 1/sqrt(...)
//         // x_frn shape: (N, C, H, W)
//
//         // Apply learnable affine transformation (gamma, beta)
//         torch::Tensor x_affine = x_frn * gamma_ + beta_;
//         // x_affine shape: (N, C, H, W)
//
//
//         // --- 2. TLU: Thresholded Linear Unit ---
//         // y = max(x_affine, tau)
//         torch::Tensor output = torch::max(x_affine, tau_);
//         // output shape: (N, C, H, W)
//
//         return output;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "FilterResponseNormalization(num_features=" << num_features_
//                << ", eps=" << eps_ << ")";
//     }
// };
// TORCH_MODULE(FilterResponseNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 64;
//     int64_t N = 4, H = 16, W = 16;
//
//     // --- Test Case 1: FRN with default parameters ---
//     std::cout << "--- Test Case 1: FRN defaults ---" << std::endl;
//     FilterResponseNormalization frn_module1(num_features);
//     // std::cout << frn_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_features, H, W});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//
//     torch::Tensor y1 = frn_module1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//     std::cout << "Output y1 mean (all): " << y1.mean().item<double>() << std::endl;
//     std::cout << "Output y1 std (all): " << y1.std().item<double>() << std::endl;
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//
//     // Check if output is non-negative after TLU (with tau=0 default) if gamma > 0
//     // Since gamma=1 (default), beta=0, tau=0, TLU is max(x_frn, 0) which is ReLU like on x_frn.
//     // x_frn can be negative.
//     // If tau is 0, output must be >= 0.
//     bool all_non_negative = (y1 >= 0).all().item<bool>();
//     std::cout << "Output y1 all non-negative (expected with default tau=0): " << (all_non_negative ? "true" : "false") << std::endl;
//     TORCH_CHECK(all_non_negative, "Output y1 should be non-negative with default tau=0.");
//
//
//     // --- Test Case 2: FRN with modified parameters ---
//     std::cout << "\n--- Test Case 2: FRN with modified parameters ---" << std::endl;
//     FilterResponseNormalization frn_module2(num_features, /*eps=*/1e-5);
//     // Modify parameters for effect
//     frn_module2->gamma_.data().fill_(-1.0); // Negative gamma
//     frn_module2->beta_.data().fill_(0.5);
//     frn_module2->tau_.data().fill_(-0.2);  // Negative tau
//
//     torch::Tensor x2 = torch::randn({N, num_features, H, W});
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//     std::cout << "gamma = " << frn_module2->gamma_.data()[0][0][0][0].item<float>()
//               << ", beta = " << frn_module2->beta_.data()[0][0][0][0].item<float>()
//               << ", tau = " << frn_module2->tau_.data()[0][0][0][0].item<float>() << std::endl;
//
//     torch::Tensor y2 = frn_module2->forward(x2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     std::cout << "Output y2 mean (all): " << y2.mean().item<double>() << std::endl;
//     // Check if values are >= tau
//     bool all_ge_tau = (y2 >= frn_module2->tau_.data()[0][0][0][0].item<float>() - 1e-6).all().item<bool>(); // Add small tolerance
//     std::cout << "Output y2 all >= tau (expected): " << (all_ge_tau ? "true" : "false") << std::endl;
//     TORCH_CHECK(all_ge_tau, "Output y2 values should be >= tau.");
//
//
//     // --- Test Case 3: Check backward pass ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     FilterResponseNormalization frn_module3(num_features);
//     frn_module3->train(); // Ensure parameters have requires_grad=true
//
//     torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
//     torch::Tensor y3 = frn_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_gamma = frn_module3->gamma_.grad().defined() &&
//                              frn_module3->gamma_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_beta = frn_module3->beta_.grad().defined() &&
//                             frn_module3->beta_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_tau = frn_module3->tau_.grad().defined() &&
//                            frn_module3->tau_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for gamma: " << (grad_exists_gamma ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for beta: " << (grad_exists_beta ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for tau: " << (grad_exists_tau ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_gamma, "No gradient for gamma!");
//     TORCH_CHECK(grad_exists_beta, "No gradient for beta!");
//     TORCH_CHECK(grad_exists_tau, "No gradient for tau!");
//
//     std::cout << "\nFilterResponseNormalization tests finished." << std::endl;
//     return 0;
// }



namespace xt::norm
{
    auto FilterResponseNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
