#include <normalizations/local_contrast_normalization.h>

//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct LocalContrastNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct LocalContrastNormalization : torch::nn::ModuleHolder<LocalContrastNormalizationImpl> {
//     using torch::nn::ModuleHolder<LocalContrastNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for LocalContrastNormalization
// struct LocalContrastNormalizationImpl : torch::nn::Module {
//     int64_t kernel_size_; // Size of the local neighborhood window (e.g., 3, 5, 7)
//     double alpha_;        // Scaling factor for the denominator (contrast term)
//     double beta_;         // Additive constant in the denominator (regularization)
//     double eps_;          // Small epsilon for sqrt to prevent NaN if variance is zero
//
//     // Pooling layers for local statistics
//     // For local mean: Average pooling
//     torch::nn::AvgPool2d local_mean_pool_{nullptr};
//     // For local variance/std dev: we need sum of squares and sum of values.
//     // Can use AvgPool2d for sum_x / N and sum_x_sq / N.
//     // Var(X) = E[X^2] - (E[X])^2
//     torch::nn::AvgPool2d local_sq_mean_pool_{nullptr}; // For E[X^2]
//
//     LocalContrastNormalizationImpl(int64_t kernel_size = 5,
//                                    double alpha = 1.0, // Often 1.0 or related to kernel_size
//                                    double beta = 0.5,  // Or a small value like 0.0001
//                                    double eps = 1e-5)
//         : kernel_size_(kernel_size),
//           alpha_(alpha),
//           beta_(beta),
//           eps_(eps) {
//
//         TORCH_CHECK(kernel_size_ > 0 && kernel_size_ % 2 == 1,
//                     "kernel_size must be a positive odd integer.");
//
//         // Padding to keep spatial dimensions the same after pooling
//         int64_t padding = (kernel_size_ - 1) / 2;
//
//         // Pool for local mean: E[X]
//         local_mean_pool_ = torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(kernel_size_).stride(1).padding(padding).count_include_pad(false)
//         );
//         register_module("local_mean_pool", local_mean_pool_);
//
//         // Pool for mean of squares: E[X^2]
//         // We'll apply this to x^2 in the forward pass.
//         local_sq_mean_pool_ = torch::nn::AvgPool2d(
//             torch::nn::AvgPool2dOptions(kernel_size_).stride(1).padding(padding).count_include_pad(false)
//         );
//         register_module("local_sq_mean_pool", local_sq_mean_pool_);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x is expected to be 4D: (N, C, H, W)
//         TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());
//
//         // --- 1. Local Mean Subtraction (Centering) ---
//         torch::Tensor local_mean_x = local_mean_pool_->forward(x); // E[X]
//         torch::Tensor x_centered = x - local_mean_x;
//
//         // --- 2. Divisive Normalization (Local Contrast Scaling) ---
//         // Calculate local variance: Var(X) = E[X^2] - (E[X])^2
//         torch::Tensor x_squared = x.pow(2);
//         torch::Tensor local_mean_x_sq = local_sq_mean_pool_->forward(x_squared); // E[X^2]
//
//         torch::Tensor local_variance = local_mean_x_sq - local_mean_x.pow(2);
//         // Ensure variance is non-negative due to potential floating point issues
//         local_variance = torch::relu(local_variance); // or local_variance.clamp_min(0)
//
//         torch::Tensor local_std_dev = torch::sqrt(local_variance + eps_); // Add eps_ inside sqrt
//
//         // The denominator term in LCN is often beta + alpha * local_std_dev
//         // Or sometimes beta is a floor for the std_dev: max(beta, alpha * local_std_dev)
//         // A common form for the denominator is: (beta + alpha * sum_weights_in_kernel * local_variance_unnormalized ) ^ power
//         // Let's use a simpler form close to Divisive Normalization:
//         // x_out = x_centered / (beta_ + alpha_ * local_std_dev)
//         // The original LCN by Jarrett et al. (CVPR'09 "What is the Best Multi-Stage Architecture for Object Recognition?")
//         // used a weighted sum for standard deviation.
//         // A common practical LCN version (e.g., in Theano or some old Caffe layers):
//         //   diff = x - local_mean
//         //   std_term = sqrt( E[(x - local_mean)^2] ) = sqrt( E[diff^2] )
//         //   pooled_sq_diff = local_mean_pool(diff^2)
//         //   denominator = max(constant, sqrt(pooled_sq_diff))
//         //   output = diff / denominator
//
//         // Let's use the Var(X) = E[X^2] - (E[X])^2 approach for std_dev.
//         torch::Tensor denominator = beta_ + alpha_ * local_std_dev;
//         // Denominator should not be too small to avoid explosion
//         // Some implementations might use torch::max(denominator, some_small_constant)
//         // For simplicity, relying on beta_ and eps_ for now.
//
//         torch::Tensor x_lcn = x_centered / denominator;
//
//         return x_lcn;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "LocalContrastNormalization(kernel_size=" << kernel_size_
//                << ", alpha=" << alpha_ << ", beta=" << beta_
//                << ", eps=" << eps_ << ")";
//     }
// };
// TORCH_MODULE(LocalContrastNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_channels = 3;
//     int64_t N = 2, H = 16, W = 16;
//     int64_t kernel_size = 5;
//
//     // --- Test Case 1: LCN with default parameters ---
//     std::cout << "--- Test Case 1: LCN defaults ---" << std::endl;
//     LocalContrastNormalization lcn_module1(kernel_size);
//     // std::cout << lcn_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_channels, H, W}) * 10 + 5; // Input with some variance and mean
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//     std::cout << "Input x1 mean (channel 0, batch 0): " << x1[0][0].mean().item<double>() << std::endl;
//     std::cout << "Input x1 std (channel 0, batch 0): " << x1[0][0].std().item<double>() << std::endl;
//
//     torch::Tensor y1 = lcn_module1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//     // After LCN, the local mean should be close to 0 and local std dev related to 1/alpha
//     // Global stats will be different.
//     std::cout << "Output y1 mean (channel 0, batch 0): " << y1[0][0].mean().item<double>() << std::endl;
//     std::cout << "Output y1 std (channel 0, batch 0): " << y1[0][0].std().item<double>() << std::endl;
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//
//
//     // --- Test Case 2: LCN with different alpha, beta ---
//     std::cout << "\n--- Test Case 2: LCN with different alpha, beta ---" << std::endl;
//     LocalContrastNormalization lcn_module2(kernel_size, /*alpha=*/0.5, /*beta=*/1.0);
//     // std::cout << lcn_module2 << std::endl;
//
//     torch::Tensor x2 = torch::ones({N, num_channels, H, W}) * 5.0; // Uniform input
//     std::cout << "Input x2 shape (uniform value 5.0): " << x2.sizes() << std::endl;
//
//     torch::Tensor y2 = lcn_module2->forward(x2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     // For uniform input: local_mean = 5, x_centered = 0. So output should be 0.
//     std::cout << "Output y2 mean (all): " << y2.mean().item<double>() << std::endl;
//     std::cout << "Output y2[0,0,H/2,W/2] (should be ~0): " << y2[0][0][H/2][W/2].item<double>() << std::endl;
//     TORCH_CHECK(torch::allclose(y2, torch::zeros_like(y2), 1e-5, 1e-7),
//                 "Output y2 for uniform input should be close to zero.");
//
//
//     // --- Test Case 3: Check backward pass (though LCN has no learnable params) ---
//     // Gradients should flow through x.
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     LocalContrastNormalization lcn_module3(kernel_size);
//     lcn_module3->train(); // Mode doesn't change LCN behavior itself
//
//     torch::Tensor x3 = torch::randn({N, num_channels, H, W}, torch::requires_grad());
//     torch::Tensor y3 = lcn_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//
//     // Check that there are no parameters to be updated
//     auto params = lcn_module3->parameters();
//     std::cout << "Number of learnable parameters: " << params.size() << std::endl;
//     TORCH_CHECK(params.empty(), "LCN should have no learnable parameters in this implementation.");
//
//
//     std::cout << "\nLocalContrastNormalization tests finished." << std::endl;
//     return 0;
// }

namespace xt::norm
{
    LocalContrastNorm::LocalContrastNorm(int64_t kernel_size,
                                         double alpha, // Often 1.0 or related to kernel_size
                                         double beta, // Or a small value like 0.0001
                                         double eps)
        : kernel_size_(kernel_size),
          alpha_(alpha),
          beta_(beta),
          eps_(eps)
    {
        TORCH_CHECK(kernel_size_ > 0 && kernel_size_ % 2 == 1,
                    "kernel_size must be a positive odd integer.");

        // Padding to keep spatial dimensions the same after pooling
        int64_t padding = (kernel_size_ - 1) / 2;

        // Pool for local mean: E[X]
        local_mean_pool_ = torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions(kernel_size_).stride(1).padding(padding).count_include_pad(false)
        );
        register_module("local_mean_pool", local_mean_pool_);

        // Pool for mean of squares: E[X^2]
        // We'll apply this to x^2 in the forward pass.
        local_sq_mean_pool_ = torch::nn::AvgPool2d(
            torch::nn::AvgPool2dOptions(kernel_size_).stride(1).padding(padding).count_include_pad(false)
        );
        register_module("local_sq_mean_pool", local_sq_mean_pool_);
    }

    auto LocalContrastNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);

        // Input x is expected to be 4D: (N, C, H, W)
        TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());

        // --- 1. Local Mean Subtraction (Centering) ---
        torch::Tensor local_mean_x = local_mean_pool_->forward(x); // E[X]
        torch::Tensor x_centered = x - local_mean_x;

        // --- 2. Divisive Normalization (Local Contrast Scaling) ---
        // Calculate local variance: Var(X) = E[X^2] - (E[X])^2
        torch::Tensor x_squared = x.pow(2);
        torch::Tensor local_mean_x_sq = local_sq_mean_pool_->forward(x_squared); // E[X^2]

        torch::Tensor local_variance = local_mean_x_sq - local_mean_x.pow(2);
        // Ensure variance is non-negative due to potential floating point issues
        local_variance = torch::relu(local_variance); // or local_variance.clamp_min(0)

        torch::Tensor local_std_dev = torch::sqrt(local_variance + eps_); // Add eps_ inside sqrt

        // The denominator term in LCN is often beta + alpha * local_std_dev
        // Or sometimes beta is a floor for the std_dev: max(beta, alpha * local_std_dev)
        // A common form for the denominator is: (beta + alpha * sum_weights_in_kernel * local_variance_unnormalized ) ^ power
        // Let's use a simpler form close to Divisive Normalization:
        // x_out = x_centered / (beta_ + alpha_ * local_std_dev)
        // The original LCN by Jarrett et al. (CVPR'09 "What is the Best Multi-Stage Architecture for Object Recognition?")
        // used a weighted sum for standard deviation.
        // A common practical LCN version (e.g., in Theano or some old Caffe layers):
        //   diff = x - local_mean
        //   std_term = sqrt( E[(x - local_mean)^2] ) = sqrt( E[diff^2] )
        //   pooled_sq_diff = local_mean_pool(diff^2)
        //   denominator = max(constant, sqrt(pooled_sq_diff))
        //   output = diff / denominator

        // Let's use the Var(X) = E[X^2] - (E[X])^2 approach for std_dev.
        torch::Tensor denominator = beta_ + alpha_ * local_std_dev;
        // Denominator should not be too small to avoid explosion
        // Some implementations might use torch::max(denominator, some_small_constant)
        // For simplicity, relying on beta_ and eps_ for now.

        torch::Tensor x_lcn = x_centered / denominator;

        return x_lcn;
    }
}
