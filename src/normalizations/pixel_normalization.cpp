#include "include/normalizations/pixel_normalization.h"


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <cmath> // For std::sqrt
//
// // Forward declaration for the Impl struct
// struct PixelNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct PixelNormalization : torch::nn::ModuleHolder<PixelNormalizationImpl> {
//     using torch::nn::ModuleHolder<PixelNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for PixelNormalization
// struct PixelNormalizationImpl : torch::nn::Module {
//     double eps_; // Small epsilon for numerical stability
//
//     PixelNormalizationImpl(double eps = 1e-8)
//         : eps_(eps) {
//         // PixelNorm typically does not have learnable parameters.
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x is expected to be 4D: (N, C, H, W)
//         TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());
//
//         // --- PixelNorm Calculation ---
//         // For each pixel (h,w), normalize the C-dimensional feature vector.
//         // x_chw = x_chw / sqrt(mean(x_c'hw^2 over c') + eps)
//
//         // 1. Square the input: x^2
//         torch::Tensor x_squared = x.pow(2); // Shape (N, C, H, W)
//
//         // 2. Calculate mean of squares across the channel dimension (dim=1)
//         // Keepdim=true to maintain shape for broadcasting.
//         torch::Tensor mean_sq_across_channels = x_squared.mean(/*dim=*/1, /*keepdim=*/true);
//         // mean_sq_across_channels shape: (N, 1, H, W)
//
//         // 3. Calculate the normalization factor: 1.0 / sqrt(mean_sq_across_channels + eps)
//         // This is equivalent to rsqrt(mean_sq_across_channels + eps)
//         torch::Tensor norm_factor = torch::rsqrt(mean_sq_across_channels + eps_);
//         // norm_factor shape: (N, 1, H, W)
//
//         // 4. Multiply the original input x by the normalization factor.
//         // The norm_factor will broadcast across the channel dimension.
//         torch::Tensor output = x * norm_factor;
//         // output shape: (N, C, H, W)
//
//         return output;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "PixelNormalization(eps=" << eps_ << ")";
//     }
// };
// TORCH_MODULE(PixelNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_channels = 3;
//     int64_t N = 2, H = 4, W = 4;
//
//     // --- Test Case 1: PixelNorm with default epsilon ---
//     std::cout << "--- Test Case 1: PixelNorm defaults ---" << std::endl;
//     PixelNormalization pixelnorm_module1; // eps = 1e-8 by default
//     // std::cout << pixelnorm_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_channels, H, W});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//
//     // Calculate L2 norm of the feature vector at a pixel before PixelNorm
//     // For x1[n, :, h, w], its L2 norm is sqrt(sum(x1[n, c, h, w]^2 over c))
//     auto pixel_vector_before = x1.select(0,0).select(2,0).select(3,0); // x1[0, :, 0, 0]
//     double l2_norm_before_pixel = std::sqrt(pixel_vector_before.pow(2).sum().item<double>());
//     // The PixelNorm formula normalizes by sqrt(mean_sq), which is sqrt( (1/C) * sum_sq ).
//     // So, an output vector y_pixel will have sum(y_pixel^2) = sum( (x_pixel / sqrt(mean_sq_x))^2 )
//     // = sum(x_pixel^2) / mean_sq_x = sum(x_pixel^2) / ( (1/C) * sum(x_pixel^2) ) = C.
//     // So, sqrt(sum(y_pixel^2)) = sqrt(C).
//     std::cout << "L2 norm of x1[0,:,0,0] before PixelNorm: " << l2_norm_before_pixel << std::endl;
//
//
//     torch::Tensor y1 = pixelnorm_module1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//
//     // Check the L2 norm of the feature vector at a pixel after PixelNorm.
//     // If y = x / sqrt(mean(x^2 across C) + eps),
//     // then sum_c(y_c^2) = sum_c(x_c^2 / (mean(x^2 across C) + eps))
//     //                = (sum_c(x_c^2)) / ( (1/C) * sum_c(x_c^2) + eps )
//     // If eps is small, this is approx. C.
//     // So, sqrt(sum_c(y_c^2)) should be approx. sqrt(C).
//     auto pixel_vector_after = y1.select(0,0).select(2,0).select(3,0); // y1[0, :, 0, 0]
//     double l2_norm_after_pixel = std::sqrt(pixel_vector_after.pow(2).sum().item<double>());
//     double expected_l2_norm_after = std::sqrt(static_cast<double>(num_channels));
//
//     std::cout << "L2 norm of y1[0,:,0,0] after PixelNorm: " << l2_norm_after_pixel
//               << " (expected approx. sqrt(C) = " << expected_l2_norm_after << ")" << std::endl;
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//     TORCH_CHECK(std::abs(l2_norm_after_pixel - expected_l2_norm_after) < 1e-5,
//                 "L2 norm of pixel vector after PixelNorm is not sqrt(C).");
//
//
//     // --- Test Case 2: Input with some structure ---
//     std::cout << "\n--- Test Case 2: Input with some structure ---" << std::endl;
//     PixelNormalization pixelnorm_module2(1e-6);
//     torch::Tensor x2 = torch::ones({N, num_channels, H, W}) * 2.0; // All elements are 2.0
//     std::cout << "Input x2 shape: " << x2.sizes() << ", all elements are 2.0" << std::endl;
//
//     // For x2[n,c,h,w] = 2.0:
//     // x_squared = 4.0
//     // mean_sq_across_channels = mean({4.0, 4.0, ... C times}) = 4.0
//     // norm_factor = 1.0 / sqrt(4.0 + eps) approx 1.0 / 2.0 = 0.5
//     // output = x * norm_factor = 2.0 * 0.5 = 1.0
//     torch::Tensor y2 = pixelnorm_module2->forward(x2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     std::cout << "Output y2[0,0,0,0] (should be ~1.0): " << y2[0][0][0][0].item<double>() << std::endl;
//     TORCH_CHECK(torch::allclose(y2, torch::ones_like(y2), 1e-5, 1e-7),
//                 "Output y2 for uniform input is not close to 1.0.");
//
//
//     // --- Test Case 3: Check backward pass (PixelNorm has no learnable params) ---
//     // Gradients should flow through x.
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     PixelNormalization pixelnorm_module3;
//     pixelnorm_module3->train(); // Mode doesn't change PixelNorm behavior
//
//     torch::Tensor x3 = torch::randn({N, num_channels, H, W}, torch::requires_grad());
//     torch::Tensor y3 = pixelnorm_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//
//     auto params = pixelnorm_module3->parameters();
//     std::cout << "Number of learnable parameters: " << params.size() << std::endl;
//     TORCH_CHECK(params.empty(), "PixelNorm should have no learnable parameters.");
//
//
//     std::cout << "\nPixelNormalization tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    PixelNorm::PixelNorm(double eps)
        : eps_(eps)
    {
        // PixelNorm typically does not have learnable parameters.
    }

    auto PixelNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);

        // Input x is expected to be 4D: (N, C, H, W)
        TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());

        // --- PixelNorm Calculation ---
        // For each pixel (h,w), normalize the C-dimensional feature vector.
        // x_chw = x_chw / sqrt(mean(x_c'hw^2 over c') + eps)

        // 1. Square the input: x^2
        torch::Tensor x_squared = x.pow(2); // Shape (N, C, H, W)

        // 2. Calculate mean of squares across the channel dimension (dim=1)
        // Keepdim=true to maintain shape for broadcasting.
        torch::Tensor mean_sq_across_channels = x_squared.mean(/*dim=*/1, /*keepdim=*/true);
        // mean_sq_across_channels shape: (N, 1, H, W)

        // 3. Calculate the normalization factor: 1.0 / sqrt(mean_sq_across_channels + eps)
        // This is equivalent to rsqrt(mean_sq_across_channels + eps)
        torch::Tensor norm_factor = torch::rsqrt(mean_sq_across_channels + eps_);
        // norm_factor shape: (N, 1, H, W)

        // 4. Multiply the original input x by the normalization factor.
        // The norm_factor will broadcast across the channel dimension.
        torch::Tensor output = x * norm_factor;
        // output shape: (N, C, H, W)

        return output;
    }
}
