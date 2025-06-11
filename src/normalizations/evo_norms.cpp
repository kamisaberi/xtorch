#include "include/normalizations/evo_norms.h"

// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <cmath> // For std::sqrt, std::max
//
// // Forward declaration for the Impl struct
// struct EvoNormS0Impl;
//
// // The main module struct that users will interact with.
// struct EvoNormS0 : torch::nn::ModuleHolder<EvoNormS0Impl> {
//     using torch::nn::ModuleHolder<EvoNormS0Impl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for EvoNormS0
// struct EvoNormS0Impl : torch::nn::Module {
//     int64_t num_features_; // Number of channels C
//     int64_t groups_;       // Number of groups to divide channels into for std computation
//     double eps_;
//     bool apply_bias_;      // Whether to add a learnable bias at the end
//
//     // Learnable parameters
//     torch::Tensor v_;      // Per-channel learnable vector 'v'
//     torch::Tensor beta_;   // Optional per-channel learnable bias (if apply_bias_ is true)
//     // Note: The original paper doesn't explicitly mention a 'gamma' like in BN.
//     // 'v' serves a similar scaling role but non-linearly within the sigmoid.
//
//     EvoNormS0Impl(int64_t num_features, int64_t groups = 32, double eps = 1e-5, bool apply_bias = false)
//         : num_features_(num_features),
//           groups_(groups),
//           eps_(eps),
//           apply_bias_(apply_bias) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//         TORCH_CHECK(groups > 0, "groups must be positive.");
//         TORCH_CHECK(num_features % groups_ == 0, "num_features must be divisible by groups.");
//
//         // Initialize learnable parameter 'v'
//         // Often initialized to ones or with some specific scheme. Let's use ones.
//         v_ = register_parameter("v", torch::ones({1, num_features_, 1, 1})); // Shape for broadcasting with NCHW
//
//         if (apply_bias_) {
//             beta_ = register_parameter("beta", torch::zeros({1, num_features_, 1, 1})); // Shape for broadcasting
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x is expected to be 4D: (N, C, H, W)
//         // C must be num_features_
//         TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
//
//         int64_t N = x.size(0);
//         int64_t C = x.size(1); // num_features_
//         int64_t H = x.size(2);
//         int64_t W = x.size(3);
//
//         int64_t channels_per_group = C / groups_;
//
//         // --- Compute Group Standard Deviation ---
//         // Reshape x to (N, groups, channels_per_group, H, W) to compute std over (channels_per_group, H, W)
//         torch::Tensor x_reshaped_for_std = x.view({N, groups_, channels_per_group, H, W});
//
//         // Calculate variance over dimensions (channels_per_group, H, W) which are dims 2, 3, 4
//         // keepdim=true to maintain shape for broadcasting
//         // unbiased=false for population variance, usually preferred
//         torch::Tensor group_var = x_reshaped_for_std.var(std::vector<int64_t>{2, 3, 4}, /*unbiased=*/false, /*keepdim=*/true);
//         // group_var shape: (N, groups_, 1, 1, 1)
//
//         torch::Tensor group_std = torch::sqrt(group_var + eps_); // Shape: (N, groups_, 1, 1, 1)
//
//         // Reshape group_std back to be broadcastable with original x shape (N, C, H, W)
//         // We need to repeat the std dev for each channel within its group.
//         // group_std: (N, groups_, 1, 1, 1) -> (N, groups_, 1, 1, 1, 1) -> tile -> (N, groups, channels_per_group, 1, 1)
//         // -> reshape to (N, C, 1, 1)
//         // A simpler way: reshape to (N, groups, 1, 1, 1) then use repeat_interleave or careful view+expand
//         // Or, more directly:
//         group_std = group_std.view({N, groups_, 1, 1, 1}); // Ensure it's 5D for tile/repeat
//         // For each N, G, want to expand out to C/G for the channel dim
//         // This is equivalent to PyTorch's `std.reshape(N, self.groups, 1, 1, 1).repeat(1, 1, C // self.groups, H, W).reshape(N, C, H, W)`
//         // Simplified if we just need to divide x by it:
//         // We can reshape group_std to (N, groups_, 1) and x to (N, groups_, channels_per_group * H * W)
//         // then divide, then reshape x back.
//         // Or, use `torch::nn::functional::group_norm`'s way of doing std calculation if available & suitable.
//         // Let's make `group_std` broadcast correctly with `x` of shape (N, C, H, W).
//         // Original `group_std` is `(N, groups, 1, 1, 1)`. We want `(N, C, 1, 1)` effectively.
//         // Each of the `groups` std values should apply to `channels_per_group` channels.
//         torch::Tensor group_std_broadcastable = group_std.repeat_interleave(channels_per_group, /*dim=*/1); // Repeats along group dim effectively
//                                                                                                         // but we want to expand it to match C channels
//         // Reshape group_std: (N, groups, 1, 1, 1)
//         // We need to make it (N, C, 1, 1) where for each group g, its std value is used for all channels_per_group in that group.
//         group_std = group_std.reshape({N, groups_, 1}); // (N, G, 1)
//         group_std = group_std.repeat_interleave(channels_per_group, /*dim=*/1); // (N, C, 1)
//         group_std = group_std.view({N, C, 1, 1}); // (N, C, 1, 1) to broadcast with (N,C,H,W)
//
//
//         // --- EvoNorm-S0 Calculation ---
//         // Numerator: v * x
//         // Denominator: group_std
//         // sigmoid_arg = (v * x) / group_std  --- this is one interpretation
//         // OR sigmoid_arg = v * x , and then y = x * sigmoid(sigmoid_arg) / group_std  --- another common interpretation.
//         // The paper's figure 2 formula: y = x * sigmoid(v*x / max(std, eps))
//         // So, let's follow that:
//         torch::Tensor numerator = v_ * x; // v_ is (1,C,1,1), x is (N,C,H,W) -> (N,C,H,W)
//         torch::Tensor norm_input = numerator / group_std; // (N,C,H,W)
//
//         torch::Tensor y = x * torch::sigmoid(norm_input); // (N,C,H,W)
//
//         if (apply_bias_) {
//             y = y + beta_; // beta_ is (1,C,1,1)
//         }
//
//         return y;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "EvoNormS0(num_features=" << num_features_
//                << ", groups=" << groups_
//                << ", eps=" << eps_
//                << ", apply_bias=" << (apply_bias_ ? "true" : "false") << ")";
//     }
// };
// TORCH_MODULE(EvoNormS0);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 64;
//     int64_t groups = 16; // num_features must be divisible by groups
//     int64_t N = 4, H = 16, W = 16;
//
//     // --- Test Case 1: EvoNormS0 with default parameters ---
//     std::cout << "--- Test Case 1: EvoNormS0 defaults ---" << std::endl;
//     EvoNormS0 evonorm_s0_module1(num_features, groups);
//     // std::cout << evonorm_s0_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_features, H, W});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//
//     torch::Tensor y1 = evonorm_s0_module1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//     std::cout << "Output y1 mean (all): " << y1.mean().item<double>() << std::endl;
//     std::cout << "Output y1 std (all): " << y1.std().item<double>() << std::endl;
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//
//
//     // --- Test Case 2: EvoNormS0 with bias ---
//     std::cout << "\n--- Test Case 2: EvoNormS0 with bias ---" << std::endl;
//     EvoNormS0 evonorm_s0_module2(num_features, groups, /*eps=*/1e-4, /*apply_bias=*/true);
//     // std::cout << evonorm_s0_module2 << std::endl;
//     // Modify bias for effect
//     evonorm_s0_module2->beta_.data().fill_(0.5);
//
//
//     torch::Tensor x2 = torch::randn({N, num_features, H, W});
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//
//     torch::Tensor y2 = evonorm_s0_module2->forward(x2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     std::cout << "Output y2 mean (all, with bias=0.5): " << y2.mean().item<double>() << std::endl;
//     TORCH_CHECK(y2.sizes() == x2.sizes(), "Output y2 shape mismatch!");
//
//
//     // --- Test Case 3: Check backward pass ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     EvoNormS0 evonorm_s0_module3(num_features, groups, 1e-5, true);
//     evonorm_s0_module3->train(); // Ensure parameters have requires_grad=true
//
//     torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
//     torch::Tensor y3 = evonorm_s0_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_v = evonorm_s0_module3->v_.grad().defined() &&
//                          evonorm_s0_module3->v_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_beta = evonorm_s0_module3->beta_.grad().defined() &&
//                             evonorm_s0_module3->beta_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for v: " << (grad_exists_v ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for beta: " << (grad_exists_beta ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_v, "No gradient for v!");
//     TORCH_CHECK(grad_exists_beta, "No gradient for beta!");
//
//     // Test with groups = 1 (equivalent to InstanceNorm std for the denominator)
//     std::cout << "\n--- Test Case 4: EvoNormS0 with groups=1 ---" << std::endl;
//     EvoNormS0 evonorm_s0_module4(num_features, /*groups=*/1);
//     torch::Tensor y4 = evonorm_s0_module4->forward(x1);
//     std::cout << "Output y4 (groups=1) shape: " << y4.sizes() << std::endl;
//     TORCH_CHECK(y4.sizes() == x1.sizes(), "Output y4 shape mismatch!");
//
//
//     std::cout << "\nEvoNormS0 tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    auto EvoNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
