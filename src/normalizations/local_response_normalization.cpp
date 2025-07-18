#include <normalizations/local_response_normalization.h>

//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <cmath> // For std::pow
//
// // Forward declaration for the Impl struct
// struct LocalResponseNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct LocalResponseNormalization : torch::nn::ModuleHolder<LocalResponseNormalizationImpl> {
//     using torch::nn::ModuleHolder<LocalResponseNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for LocalResponseNormalization
// struct LocalResponseNormalizationImpl : torch::nn::Module {
//     int64_t size_;     // `n` in the formula: the number of channels to sum over for normalization.
//     double alpha_;     // Scaling parameter `alpha`.
//     double beta_;      // Exponent `beta`.
//     double k_;         // Additive constant `k`.
//
//     // For implementing the sum over channels, we can use a "channel-wise" pooling approach.
//     // PyTorch's nn.LocalResponseNorm uses an internal C++ implementation.
//     // We can simulate it using unfold/fold or a carefully constructed Conv1d/AvgPool1d on channels if needed.
//     // However, a more direct approach by iterating or using slicing and sum is also possible for clarity,
//     // though potentially less optimized than a fused kernel.
//     // For this implementation, we will use a direct computation that iterates through channels conceptually,
//     // which can be achieved efficiently with tensor operations.
//
//     LocalResponseNormalizationImpl(int64_t size, double alpha = 1e-4, double beta = 0.75, double k = 1.0)
//         : size_(size),
//           alpha_(alpha),
//           beta_(beta),
//           k_(k) {
//         TORCH_CHECK(size_ > 0 && size_ % 2 == 1, "LRN size must be a positive odd integer.");
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x is expected to be 4D: (N, C, H, W)
//         TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());
//
//         int64_t N = x.size(0);
//         int64_t C = x.size(1);
//         int64_t H = x.size(2);
//         int64_t W = x.size(3);
//
//         // --- LRN Calculation ---
//         // Denominator: (k + alpha/n * sum_neighbors(a^2) )^beta
//         // Note: PyTorch's nn.LocalResponseNorm uses alpha * sum(...) instead of alpha/n * sum(...)
//         // The original AlexNet paper divides alpha by n. We'll follow PyTorch's nn.LocalResponseNorm convention
//         // which does NOT divide alpha by n in its formula: (k + alpha * sum(...))^beta.
//         // If you need the division by n for alpha, you'd adjust `alpha_` at construction or here.
//         // Let's stick to the PyTorch's nn.LocalResponseNorm formula:
//         // b_i = a_i / (k + (alpha / size) * sum_{j in window} a_j^2 ) ^ beta
//         // The `alpha / size` part is what nn.LocalResponseNorm does internally if `alpha` is given as input.
//         // Or, more directly like Krizhevsky paper: (k + alpha * sum a_j^2)^beta,
//         // where alpha is small (e.g., 1e-4) and n is the sum window size.
//         // Let's use the Krizhevsky formula with `alpha_` directly.
//         // If PyTorch's `nn.LocalResponseNorm` is the target, then the sum is over `size_` channels,
//         // and the `alpha` parameter to `nn.LocalResponseNorm` gets internally divided by `size_`.
//         // To match PyTorch's nn.LocalResponseNorm behavior given its parameters:
//         // actual_alpha_for_sum = this->alpha_ / this->size_;
//         // But the common formula has alpha directly multiplying the sum.
//         // Let's use `div_term = x.pow(2).unsqueeze(1)` and then a 1D average pool on channel dim.
//
//         // Create a tensor for the sum of squares.
//         // We need to sum over a window of `size_` channels.
//         // This can be done efficiently using a 1D average pooling operation on the squared input,
//         // applied across the channel dimension.
//         // We need to treat (N,C,H,W) as (N*H*W, C) then pool, then reshape. Or use Conv3D with 1xSIZE_x1x1 kernel.
//         // PyTorch's nn.LocalResponseNorm does this more directly.
//
//         // Simpler direct approach for clarity (can be optimized with conv/pool):
//         torch::Tensor x_squared = x.pow(2);
//         torch::Tensor sum_sq = torch::zeros_like(x);
//
//         // Half window size for channels
//         int radius = (size_ - 1) / 2;
//
//         // Pad x_squared along channel dimension for easier window sum
//         // Pad with (radius, radius) on dimension 1 (channels)
//         auto x_squared_padded = torch::constant_pad_nd(x_squared, {0,0, 0,0, radius,radius, 0,0}, 0.0);
//
//         // Iterate and sum (conceptually, implemented with slicing)
//         for (int64_t i = 0; i < C; ++i) {
//             // Slice the padded tensor to get the window for channel i
//             // The window for original channel `i` is from `i` to `i + size_ -1` in the padded tensor.
//             sum_sq.select(1, i) = x_squared_padded.slice(1, i, i + size_).sum(1);
//         }
//         // This direct sum might be slightly different from a pooling average if edge effects are handled differently.
//         // PyTorch's LRN uses a sum, and `alpha` is usually scaled by `size_`.
//         // If we want to match torch.nn.functional.local_response_norm exactly:
//         // It's `x_sq = x.pow(2)`
//         // `x_sq_sum = torch.empty_like(x_sq)`
//         // `for c in range(C):`
//         // `  begin = max(0, c - size // 2)`
//         // `  end = min(C, c + size // 2 + 1)`
//         // `  x_sq_sum[:, c] = x_sq[:, begin:end].sum(dim=1)`
//         // `norm_val = k_ + (alpha_ / size_) * x_sq_sum` (This is if alpha is scaled by size)
//         // `norm_val = norm_val.pow(beta_)`
//         // `return x / norm_val`
//
//         // Let's implement the sum more directly using torch::nn::functional::avg_pool1d as a trick
//         // Reshape x_squared to (N*H*W, 1, C) to use 1D pooling as a sliding window sum
//         // Or (N, H*W, C) then permute to (N, C, H*W)
//         auto x_sq_reshaped = x_squared.permute({0, 2, 3, 1}).contiguous(); // (N, H, W, C)
//         x_sq_reshaped = x_sq_reshaped.view({-1, C}); // (N*H*W, C)
//         x_sq_reshaped = x_sq_reshaped.unsqueeze(1); // (N*H*W, 1, C) for AvgPool1d (expects N, Cin, Lin)
//
//         // Using AvgPool1d to simulate sum pooling: avg_pool * kernel_size = sum_pool
//         // Padding for AvgPool1d: kernel_size_ must be odd. padding = (kernel_size_ - 1) / 2
//         // This is a common way to implement LRN's sum.
//         auto avg_pool_op = torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(size_)
//                                                 .stride(1)
//                                                 .padding(radius)
//                                                 .count_include_pad(false)); // Important for edges if not using custom padding
//                                                                           // Krizhevsky's LRN effectively zero-pads.
//
//         torch::Tensor sum_pooled_sq = avg_pool_op(x_sq_reshaped) * size_; // (N*H*W, 1, C)
//         sum_pooled_sq = sum_pooled_sq.squeeze(1); // (N*H*W, C)
//         sum_pooled_sq = sum_pooled_sq.view({N, H, W, C});
//         sum_pooled_sq = sum_pooled_sq.permute({0, 3, 1, 2}).contiguous(); // (N, C, H, W)
//
//         // Denominator calculation
//         // The parameter `alpha` in PyTorch's nn.LocalResponseNorm corresponds to `alpha_param / size_`
//         // where `alpha_param` is the one you pass to the constructor.
//         // So, if we want to match, use `alpha_ / size_`.
//         // If following original paper more directly (where alpha is small):
//         torch::Tensor denominator = sum_pooled_sq * (alpha_ /* / size_ : if matching torch.nn.LRN precisely with its alpha meaning */) + k_;
//         denominator = denominator.pow(beta_);
//
//         torch::Tensor output = x / denominator;
//
//         return output;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "LocalResponseNormalization(size=" << size_
//                << ", alpha=" << alpha_ << ", beta=" << beta_
//                << ", k=" << k_ << ")";
//         stream << "\n  (Note: alpha might be interpreted as alpha/size by some frameworks like PyTorch's nn.LocalResponseNorm)";
//     }
// };
// TORCH_MODULE(LocalResponseNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_channels = 10; // C
//     int64_t N = 2, H = 8, W = 8;
//     int64_t lrn_size = 5;       // n
//     double lrn_alpha = 1e-4; // alpha
//     double lrn_beta = 0.75;   // beta
//     double lrn_k = 2.0;       // k (AlexNet used k=2, but PyTorch default is k=1)
//
//     // --- Test Case 1: LRN with AlexNet-like parameters ---
//     std::cout << "--- Test Case 1: LRN with AlexNet-like parameters ---" << std::endl;
//     LocalResponseNormalization lrn_module1(lrn_size, lrn_alpha, lrn_beta, lrn_k);
//     // std::cout << lrn_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_channels, H, W});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//     // std::cout << "x1 before LRN (example value): " << x1[0][0][0][0].item<double>() << std::endl;
//
//     torch::Tensor y1 = lrn_module1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//     // std::cout << "y1 after LRN (example value): " << y1[0][0][0][0].item<double>() << std::endl;
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//
//     // Compare with PyTorch's functional LRN for one sample (if possible and for validation)
//     // Note: PyTorch's nn.LocalResponseNorm constructor's alpha is effectively alpha_paper / size.
//     // So, to match, if our `lrn_alpha` is `alpha_paper`, then `alpha_pytorch = lrn_alpha / lrn_size`.
//     // The `forward` here currently uses `alpha_` directly in `alpha_ * sum_sq`.
//     // If we want to match `torch.nn.functional.local_response_norm` which takes `alpha` that it then divides by `size`:
//     // The sum in `torch.nn.functional.local_response_norm` is over `size` elements.
//     // Its formula is `x / (k + (alpha_param / size) * sum_sq_window).pow(beta)`
//     // Our implementation currently is `x / (k + alpha_constructor * sum_sq_window).pow(beta)`
//     // To match `torch.nn.LocalResponseNorm(size, alpha_pytorch, beta, k)`:
//     // we would need `lrn_module1_pytorch_equivalent(size, alpha_pytorch * size, beta, k)`
//     // or change the internal formula to `sum_pooled_sq * (alpha_ / size_)`.
//     // Let's assume our current `alpha_` is the direct multiplier for now.
//
//     // For a more robust check, let's verify on a simple case.
//     // If x is all ones, and size = 1, then sum_sq = 1.
//     // Denominator = (k + alpha * 1)^beta. Output = 1 / Denominator.
//     std::cout << "\n--- Test Case 2: Simple LRN with size=1 ---" << std::endl;
//     LocalResponseNormalization lrn_module_simple(1, lrn_alpha, lrn_beta, lrn_k);
//     torch::Tensor x_ones = torch::ones({1, 1, 1, 1}); // N=1, C=1, H=1, W=1
//     torch::Tensor y_simple = lrn_module_simple.forward(x_ones);
//     double expected_denom_simple = std::pow(lrn_k + lrn_alpha * 1.0, lrn_beta);
//     double expected_y_simple = 1.0 / expected_denom_simple;
//     std::cout << "Output y_simple (for x=1, C=1, size=1): " << y_simple.item<double>() << std::endl;
//     std::cout << "Expected y_simple: " << expected_y_simple << std::endl;
//     TORCH_CHECK(std::abs(y_simple.item<double>() - expected_y_simple) < 1e-6, "Simple LRN case failed.");
//
//
//     // --- Test Case 3: Check backward pass (LRN has no learnable params) ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     LocalResponseNormalization lrn_module3(lrn_size, lrn_alpha, lrn_beta, lrn_k);
//     lrn_module3->train(); // Mode doesn't change LRN behavior
//
//     torch::Tensor x3 = torch::randn({N, num_channels, H, W}, torch::requires_grad());
//     torch::Tensor y3 = lrn_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//
//     auto params = lrn_module3->parameters();
//     std::cout << "Number of learnable parameters: " << params.size() << std::endl;
//     TORCH_CHECK(params.empty(), "LRN should have no learnable parameters.");
//
//
//     std::cout << "\nLocalResponseNormalization tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    LocalResponseNorm::LocalResponseNorm(int64_t size, double alpha, double beta, double k)
        : size_(size),
          alpha_(alpha),
          beta_(beta),
          k_(k)
    {
        TORCH_CHECK(size_ > 0 && size_ % 2 == 1, "LRN size must be a positive odd integer.");
    }

    auto LocalResponseNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);

        // Input x is expected to be 4D: (N, C, H, W)
        TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());

        int64_t N = x.size(0);
        int64_t C = x.size(1);
        int64_t H = x.size(2);
        int64_t W = x.size(3);

        // --- LRN Calculation ---
        // Denominator: (k + alpha/n * sum_neighbors(a^2) )^beta
        // Note: PyTorch's nn.LocalResponseNorm uses alpha * sum(...) instead of alpha/n * sum(...)
        // The original AlexNet paper divides alpha by n. We'll follow PyTorch's nn.LocalResponseNorm convention
        // which does NOT divide alpha by n in its formula: (k + alpha * sum(...))^beta.
        // If you need the division by n for alpha, you'd adjust `alpha_` at construction or here.
        // Let's stick to the PyTorch's nn.LocalResponseNorm formula:
        // b_i = a_i / (k + (alpha / size) * sum_{j in window} a_j^2 ) ^ beta
        // The `alpha / size` part is what nn.LocalResponseNorm does internally if `alpha` is given as input.
        // Or, more directly like Krizhevsky paper: (k + alpha * sum a_j^2)^beta,
        // where alpha is small (e.g., 1e-4) and n is the sum window size.
        // Let's use the Krizhevsky formula with `alpha_` directly.
        // If PyTorch's `nn.LocalResponseNorm` is the target, then the sum is over `size_` channels,
        // and the `alpha` parameter to `nn.LocalResponseNorm` gets internally divided by `size_`.
        // To match PyTorch's nn.LocalResponseNorm behavior given its parameters:
        // actual_alpha_for_sum = this->alpha_ / this->size_;
        // But the common formula has alpha directly multiplying the sum.
        // Let's use `div_term = x.pow(2).unsqueeze(1)` and then a 1D average pool on channel dim.

        // Create a tensor for the sum of squares.
        // We need to sum over a window of `size_` channels.
        // This can be done efficiently using a 1D average pooling operation on the squared input,
        // applied across the channel dimension.
        // We need to treat (N,C,H,W) as (N*H*W, C) then pool, then reshape. Or use Conv3D with 1xSIZE_x1x1 kernel.
        // PyTorch's nn.LocalResponseNorm does this more directly.

        // Simpler direct approach for clarity (can be optimized with conv/pool):
        torch::Tensor x_squared = x.pow(2);
        torch::Tensor sum_sq = torch::zeros_like(x);

        // Half window size for channels
        int radius = (size_ - 1) / 2;

        // Pad x_squared along channel dimension for easier window sum
        // Pad with (radius, radius) on dimension 1 (channels)
        auto x_squared_padded = torch::constant_pad_nd(x_squared, {0, 0, 0, 0, radius, radius, 0, 0}, 0.0);

        // Iterate and sum (conceptually, implemented with slicing)
        for (int64_t i = 0; i < C; ++i)
        {
            // Slice the padded tensor to get the window for channel i
            // The window for original channel `i` is from `i` to `i + size_ -1` in the padded tensor.
            sum_sq.select(1, i) = x_squared_padded.slice(1, i, i + size_).sum(1);
        }
        // This direct sum might be slightly different from a pooling average if edge effects are handled differently.
        // PyTorch's LRN uses a sum, and `alpha` is usually scaled by `size_`.
        // If we want to match torch.nn.functional.local_response_norm exactly:
        // It's `x_sq = x.pow(2)`
        // `x_sq_sum = torch.empty_like(x_sq)`
        // `for c in range(C):`
        // `  begin = max(0, c - size // 2)`
        // `  end = min(C, c + size // 2 + 1)`
        // `  x_sq_sum[:, c] = x_sq[:, begin:end].sum(dim=1)`
        // `norm_val = k_ + (alpha_ / size_) * x_sq_sum` (This is if alpha is scaled by size)
        // `norm_val = norm_val.pow(beta_)`
        // `return x / norm_val`

        // Let's implement the sum more directly using torch::nn::functional::avg_pool1d as a trick
        // Reshape x_squared to (N*H*W, 1, C) to use 1D pooling as a sliding window sum
        // Or (N, H*W, C) then permute to (N, C, H*W)
        auto x_sq_reshaped = x_squared.permute({0, 2, 3, 1}).contiguous(); // (N, H, W, C)
        x_sq_reshaped = x_sq_reshaped.view({-1, C}); // (N*H*W, C)
        x_sq_reshaped = x_sq_reshaped.unsqueeze(1); // (N*H*W, 1, C) for AvgPool1d (expects N, Cin, Lin)

        // Using AvgPool1d to simulate sum pooling: avg_pool * kernel_size = sum_pool
        // Padding for AvgPool1d: kernel_size_ must be odd. padding = (kernel_size_ - 1) / 2
        // This is a common way to implement LRN's sum.
        auto avg_pool_op = torch::nn::AvgPool1d(torch::nn::AvgPool1dOptions(size_)
                                                .stride(1)
                                                .padding(radius)
                                                .count_include_pad(false));
        // Important for edges if not using custom padding
        // Krizhevsky's LRN effectively zero-pads.

        torch::Tensor sum_pooled_sq = avg_pool_op(x_sq_reshaped) * size_; // (N*H*W, 1, C)
        sum_pooled_sq = sum_pooled_sq.squeeze(1); // (N*H*W, C)
        sum_pooled_sq = sum_pooled_sq.view({N, H, W, C});
        sum_pooled_sq = sum_pooled_sq.permute({0, 3, 1, 2}).contiguous(); // (N, C, H, W)

        // Denominator calculation
        // The parameter `alpha` in PyTorch's nn.LocalResponseNorm corresponds to `alpha_param / size_`
        // where `alpha_param` is the one you pass to the constructor.
        // So, if we want to match, use `alpha_ / size_`.
        // If following original paper more directly (where alpha is small):
        torch::Tensor denominator = sum_pooled_sq * (alpha_
            /* / size_ : if matching torch.nn.LRN precisely with its alpha meaning */) + k_;
        denominator = denominator.pow(beta_);

        torch::Tensor output = x / denominator;

        return output;
    }
}
