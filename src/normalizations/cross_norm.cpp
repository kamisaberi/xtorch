#include "include/normalizations/cross_norm.h"


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct CrossNormImpl;
//
// // The main module struct that users will interact with.
// struct CrossNorm : torch::nn::ModuleHolder<CrossNormImpl> {
//     using torch::nn::ModuleHolder<CrossNormImpl>::ModuleHolder;
//
//     // Forward method takes the primary input x and the reference input y
//     torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& y_reference) {
//         return impl_->forward(x, y_reference);
//     }
// };
//
// // The implementation struct
// struct CrossNormImpl : torch::nn::Module {
//     int64_t num_features_;
//     double eps_;
//     bool affine_; // Whether to apply learnable affine transform to x after normalization
//
//     // Learnable parameters for x (if affine is true)
//     torch::Tensor gamma_x_; // scale for x
//     torch::Tensor beta_x_;  // shift for x
//
//     CrossNormImpl(int64_t num_features, double eps = 1e-5, bool affine = true)
//         : num_features_(num_features), eps_(eps), affine_(affine) {
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//
//         if (affine_) {
//             gamma_x_ = register_parameter("weight", torch::ones({num_features_})); // For x
//             beta_x_  = register_parameter("bias",   torch::zeros({num_features_})); // For x
//         }
//         // No running_mean or running_var needed if stats are always from y_reference.
//     }
//
//     torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& y_reference) {
//         // x: The tensor to be normalized. (N, C, D1, D2, ...)
//         // y_reference: The tensor from which normalization statistics (mean, var) are derived. (N_y, C, D1_y, D2_y, ...)
//         // C (num_features) must match between x and y_reference.
//         // N and spatial dimensions can differ. Statistics from y_reference will be broadcasted.
//         // For simplicity, we'll assume y_reference also has shape (N, C, D1_y, D2_y, ...)
//         // and we compute instance-wise stats from y and apply to corresponding instance in x.
//
//         TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got ", x.dim());
//         TORCH_CHECK(y_reference.dim() >= 2, "Reference tensor y_reference must have at least 2 dimensions (N, C, ...). Got ", y_reference.dim());
//
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Number of features (channels) in x mismatch. Expected ", num_features_,
//                     ", but got ", x.size(1));
//         TORCH_CHECK(y_reference.size(1) == num_features_,
//                     "Number of features (channels) in y_reference mismatch. Expected ", num_features_,
//                     ", but got ", y_reference.size(1));
//         TORCH_CHECK(x.size(0) == y_reference.size(0),
//                     "Batch sizes of x (", x.size(0), ") and y_reference (", y_reference.size(0), ") must match for this CrossNorm interpretation.");
//
//
//         // --- 1. Compute mean and variance from y_reference (Instance Normalization style) ---
//         torch::Tensor mean_y;
//         torch::Tensor var_y;
//
//         if (y_reference.dim() > 2) { // y_reference has spatial/sequential dimensions (N, C, D1_y, ...)
//             std::vector<int64_t> reduce_dims_y; // Dims to average over for mean/var from y_reference (D1_y, D2_y, ...)
//             for (int64_t i = 2; i < y_reference.dim(); ++i) {
//                 reduce_dims_y.push_back(i);
//             }
//             // keepdim=true for broadcasting
//             mean_y = y_reference.mean(reduce_dims_y, /*keepdim=*/true); // Shape (N_y, C, 1, 1, ...)
//             var_y  = y_reference.var(reduce_dims_y, /*unbiased=*/false, /*keepdim=*/true); // Shape (N_y, C, 1, 1, ...)
//         } else { // y_reference is 2D (N_y, C). InstanceNorm on a single point per channel results in mean=value, var=0.
//             mean_y = y_reference; // Each value is its own mean
//             var_y  = torch::zeros_like(y_reference); // Variance of a single point is 0
//              // To ensure correct broadcasting later, reshape mean_y and var_y to (N_y, C, 1, 1, ...) like structure for x
//             if (x.dim() > 2) {
//                 std::vector<int64_t> view_shape_y_stats;
//                 view_shape_y_stats.push_back(y_reference.size(0)); // N_y
//                 view_shape_y_stats.push_back(num_features_);      // C
//                 for (int64_t i = 2; i < x.dim(); ++i) { // Use x's spatial dims for shaping
//                     view_shape_y_stats.push_back(1);
//                 }
//                 mean_y = mean_y.view(view_shape_y_stats);
//                 var_y  = var_y.view(view_shape_y_stats);
//             }
//         }
//
//         // --- 2. Normalize x using statistics from y_reference ---
//         // mean_y and var_y are (N, C, 1, ..., 1) based on y_reference's spatial dims.
//         // x is (N, C, D1_x, D2_x, ...). Broadcasting should work if N and C match.
//         torch::Tensor x_normalized = (x - mean_y) / torch::sqrt(var_y + eps_);
//
//
//         // --- 3. Apply optional affine transformation to x_normalized ---
//         if (affine_) {
//             // gamma_x_ and beta_x_ are (num_features_). Reshape to (1, C, 1, 1, ...) for broadcasting.
//             std::vector<int64_t> affine_param_view_shape(x.dim(), 1);
//             affine_param_view_shape[1] = num_features_;
//
//             return x_normalized * gamma_x_.view(affine_param_view_shape) + beta_x_.view(affine_param_view_shape);
//         } else {
//             return x_normalized;
//         }
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "CrossNorm(num_features=" << num_features_
//                << ", eps=" << eps_
//                << ", affine_on_x=" << (affine_ ? "true" : "false") << ")";
//     }
// };
// TORCH_MODULE(CrossNorm);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 3;
//     int64_t N = 4;
//
//     // --- Test Case 1: 4D inputs x and y_reference, affine=true ---
//     std::cout << "--- Test Case 1: 4D inputs (NCHW), affine=true ---" << std::endl;
//     CrossNorm cross_norm_module1(num_features, /*eps=*/1e-5, /*affine=*/true);
//     // std::cout << cross_norm_module1 << std::endl;
//
//     int64_t H1 = 8, W1 = 8; // Spatial dims for x
//     int64_t H_ref1 = 6, W_ref1 = 6; // Spatial dims for y_reference (can be different)
//
//     torch::Tensor x1 = torch::randn({N, num_features, H1, W1});
//     // y_reference should have some variance for meaningful normalization
//     torch::Tensor y_ref1 = torch::randn({N, num_features, H_ref1, W_ref1}) * 2.0 + 5.0;
//
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//     std::cout << "Input y_ref1 shape: " << y_ref1.sizes() << std::endl;
//
//     // Set gamma and beta to non-default to observe their effect
//     cross_norm_module1->gamma_x_.data().fill_(1.5);
//     cross_norm_module1->beta_x_.data().fill_(0.5);
//
//     torch::Tensor output1 = cross_norm_module1->forward(x1, y_ref1);
//     std::cout << "Output1 shape: " << output1.sizes() << std::endl;
//
//     // Check mean/std of output1 for one instance, one channel.
//     // It should be roughly beta_x_ and gamma_x_ IF x1 was similar to y_ref1 before normalization.
//     // More precisely, (x1 - mean(y_ref1))/std(y_ref1) should have mean 0, std 1.
//     // Then affine: result_mean = 0*gamma + beta = beta, result_std = 1*gamma = gamma.
//     auto output1_inst0_ch0 = output1.select(0,0).select(0,0); // N=0, C=0
//     std::cout << "Output1 [0,0,:,:] mean (expected ~beta_x=" << cross_norm_module1->beta_x_[0].item<double>()
//               << "): " << output1_inst0_ch0.mean().item<double>() << std::endl;
//     std::cout << "Output1 [0,0,:,:] std (expected ~gamma_x=" << cross_norm_module1->gamma_x_[0].item<double>()
//               << "): " << output1_inst0_ch0.std(false).item<double>() << std::endl;
//
//
//     // --- Test Case 2: 2D inputs x and y_reference, affine=false ---
//     // y_reference is (N,C), so its instance stats are: mean=y_ref, var=0
//     // x_normalized = (x - y_ref) / sqrt(0 + eps) -> can be very large
//     // This case might require careful handling of eps or a different strategy for 2D y_ref.
//     // The current InstanceNorm logic for y_ref will make var_y=0.
//     std::cout << "\n--- Test Case 2: 2D inputs (NC), affine=false ---" << std::endl;
//     CrossNorm cross_norm_module2(num_features, /*eps=*/1e-5, /*affine=*/false);
//     // std::cout << cross_norm_module2 << std::endl;
//
//     torch::Tensor x2 = torch::randn({N, num_features});
//     torch::Tensor y_ref2 = torch::randn({N, num_features}) * 0.5 + 1.0;
//
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//     std::cout << "Input y_ref2 shape: " << y_ref2.sizes() << std::endl;
//
//     torch::Tensor output2 = cross_norm_module2->forward(x2, y_ref2);
//     std::cout << "Output2 shape: " << output2.sizes() << std::endl;
//     // With var_y=0 from y_ref2 (2D), output will be (x2 - y_ref2) / sqrt(eps)
//     std::cout << "Output2[0] (example): " << output2[0] << std::endl;
//     torch::Tensor expected_val_approx = (x2[0][0] - y_ref2[0][0]) / std::sqrt(1e-5);
//     std::cout << "Expected output2[0][0] approx: " << expected_val_approx.item<double>()
//               << ", Actual: " << output2[0][0].item<double>() << std::endl;
//     TORCH_CHECK(std::abs(output2[0][0].item<double>() - expected_val_approx.item<double>()) < std::abs(expected_val_approx.item<double>()*0.1),
//                 "2D y_reference case output not as expected.");
//
//
//     // --- Test Case 3: Backward pass check ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     CrossNorm cross_norm_module3(num_features, 1e-5, true);
//     cross_norm_module3->train(); // Ensure gamma_x_, beta_x_ have requires_grad=true
//
//     torch::Tensor x3 = torch::randn({N, num_features, H1, W1}, torch::requires_grad());
//     torch::Tensor y_ref3 = torch::randn({N, num_features, H_ref1, W_ref1}, torch::requires_grad()); // y_ref can also have grad
//
//     torch::Tensor output3 = cross_norm_module3->forward(x3, y_ref3);
//     torch::Tensor loss = output3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_y_ref3 = y_ref3.grad().defined() && y_ref3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_gamma_x = cross_norm_module3->gamma_x_.grad().defined() &&
//                                cross_norm_module3->gamma_x_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for y_ref3: " << (grad_exists_y_ref3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for gamma_x: " << (grad_exists_gamma_x ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_y_ref3, "No gradient for y_ref3!");
//     TORCH_CHECK(grad_exists_gamma_x, "No gradient for gamma_x!");
//
//     std::cout << "\nCrossNorm tests finished." << std::endl;
//     return 0;
// }



namespace xt::norm
{
    auto CrossNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
