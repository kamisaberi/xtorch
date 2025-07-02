#include "include/normalizations/virtual_batch_normalization.h"


//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <stdexcept> // For std::runtime_error
//
// // Forward declaration for the Impl struct
// struct VirtualBatchNormImpl;
//
// // The main module struct that users will interact with.
// struct VirtualBatchNorm : torch::nn::ModuleHolder<VirtualBatchNormImpl> {
//     using torch::nn::ModuleHolder<VirtualBatchNormImpl>::ModuleHolder;
//
//     // Forward method uses pre-computed reference statistics
//     torch::Tensor forward(torch::Tensor x);
//
//     // Method to compute and store reference statistics from a reference batch
//     void init_reference_stats(const torch::Tensor& x_ref);
// };
//
// // The implementation struct for VirtualBatchNorm
// struct VirtualBatchNormImpl : torch::nn::Module {
//     int64_t num_features_;
//     double eps_;
//     bool affine_; // Whether to apply learnable affine transform
//
//     // Learnable affine parameters (gamma and beta)
//     torch::Tensor gamma_;
//     torch::Tensor beta_;
//
//     // Buffers for storing reference statistics (mu_ref, var_ref)
//     // These are computed once from a reference batch.
//     torch::Tensor mu_ref_;
//     torch::Tensor var_ref_;
//     torch::Tensor initialized_flag_; // To check if reference stats have been set
//
//     VirtualBatchNormImpl(int64_t num_features, double eps = 1e-5, bool affine = true)
//         : num_features_(num_features),
//           eps_(eps),
//           affine_(affine) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//
//         if (affine_) {
//             gamma_ = register_parameter("weight", torch::ones({num_features_}));
//             beta_ = register_parameter("bias", torch::zeros({num_features_}));
//         }
//
//         // Initialize reference stat buffers as undefined or placeholders.
//         // They will be properly sized and filled by init_reference_stats.
//         // Registering them ensures they are part of the module's state.
//         mu_ref_ = register_buffer("mu_ref", torch::Tensor());
//         var_ref_ = register_buffer("var_ref", torch::Tensor());
//         initialized_flag_ = register_buffer("initialized_flag", torch::tensor(0, torch::kBool));
//     }
//
//     // Public method to be called once to set reference statistics
//     void init_reference_stats_impl(const torch::Tensor& x_ref) {
//         TORCH_CHECK(x_ref.defined(), "Reference batch x_ref must be defined.");
//         TORCH_CHECK(x_ref.dim() >= 2, "Reference batch x_ref must have at least 2 dimensions (N_ref, C, ...).");
//         TORCH_CHECK(x_ref.size(1) == num_features_,
//                     "Reference batch channels mismatch. Expected ", num_features_, ", got ", x_ref.size(1));
//
//         // Compute mean and variance from x_ref across batch and spatial/sequential dims
//         // Statistics are per-channel (shape: {num_features_})
//         std::vector<int64_t> reduce_dims_ref_stats;
//         reduce_dims_ref_stats.push_back(0); // Batch dimension of x_ref
//         for (int64_t i = 2; i < x_ref.dim(); ++i) { // Spatial/sequential dimensions of x_ref
//             reduce_dims_ref_stats.push_back(i);
//         }
//
//         // Reshape affine params for broadcasting (e.g., to [1, C, 1, 1] for 4D input)
//         // This shape is for mu_ref_ and var_ref_ for broadcasting with x later.
//         std::vector<int64_t> ref_stat_view_shape(x_ref.dim(), 1);
//         ref_stat_view_shape[1] = num_features_;
//
//
//         // Compute mean and variance for the reference batch
//         // These will be stored with a shape like (1, C, 1, 1) for NCHW inputs to x
//         torch::Tensor computed_mu_ref = x_ref.mean(reduce_dims_ref_stats, /*keepdim=false*/ false); // Shape (C,)
//         // For var: E[X^2] - (E[X])^2
//         torch::Tensor computed_var_ref = (x_ref - computed_mu_ref.view(ref_stat_view_shape)).pow(2).mean(reduce_dims_ref_stats, /*keepdim=false*/ false); // Shape (C,)
//
//
//         // Store them in the desired broadcastable shape (e.g., 1,C,1,1 for 4D x)
//         // To determine the broadcast shape, we need the dimensionality of the *actual input x*
//         // to the forward pass. This is a bit tricky if x_ref has different dims than x.
//         // For simplicity, let's assume x in forward() will have same rank as x_ref was intended for.
//         // If x in forward might be 4D, and x_ref was used to calculate (C,) stats,
//         // mu_ref_ and var_ref_ should be stored as (1,C,1,1) or similar.
//         // Let's assume we store them as (C,) and reshape in forward.
//         // OR, decide on a target rank (e.g. 4D for images) for ref stats.
//         // For now, store as (num_features_). Reshape in forward.
//         mu_ref_.copy_(computed_mu_ref.detach());
//         var_ref_.copy_(computed_var_ref.detach());
//
//         // Or, to be more robust for typical NCHW usage, store them directly in broadcastable shape
//         // assuming forward `x` will be 4D.
//         if (mu_ref_.numel() == 0 || mu_ref_.sizes() != ref_stat_view_shape) { // If not yet initialized or shape wrong
//              mu_ref_ = register_buffer("mu_ref", computed_mu_ref.detach().view(ref_stat_view_shape));
//              var_ref_ = register_buffer("var_ref", computed_var_ref.detach().view(ref_stat_view_shape));
//         } else {
//              mu_ref_.copy_(computed_mu_ref.detach().view(ref_stat_view_shape));
//              var_ref_.copy_(computed_var_ref.detach().view(ref_stat_view_shape));
//         }
//
//
//         initialized_flag_.fill_(true);
//         std::cout << "[VirtualBatchNorm] Reference statistics initialized." << std::endl;
//     }
//
//
//     torch::Tensor forward_impl(torch::Tensor x) {
//         TORCH_CHECK(initialized_flag_.item<bool>(),
//                     "VirtualBatchNorm reference statistics not initialized. Call init_reference_stats() first.");
//         TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
//
//         // mu_ref_ and var_ref_ are stored, e.g., as (1, C, 1, 1) if initialized with 4D x_ref.
//         // They need to broadcast with x.
//         // If mu_ref_/var_ref_ are (C,) and x is (N,C,H,W), then reshape to (1,C,1,1).
//         // Our init_reference_stats stores them in broadcastable form (e.g., 1,C,1,1 for 4D x_ref).
//         // If x.dim() is different from the dim assumed during init, this might need adjustment.
//         // For this implementation, assume mu_ref_ and var_ref_ are already in a shape
//         // like (1, C, 1, ..., 1) that matches x.dim() after channel dim.
//         TORCH_CHECK(mu_ref_.dim() == x.dim() && var_ref_.dim() == x.dim(),
//                     "Dimensionality of stored reference stats (mu_ref dim: ", mu_ref_.dim(),
//                     ") must match input x dim (", x.dim(), ") for broadcasting. "
//                     "Ensure init_reference_stats was called with x_ref of appropriate dimensionality "
//                     "or adjust reshaping. Current mu_ref shape: ", mu_ref_.sizes());
//
//
//         // Normalize x using the stored reference statistics mu_ref_ and var_ref_
//         torch::Tensor x_normalized = (x - mu_ref_) / torch::sqrt(var_ref_ + eps_);
//
//         // Apply learnable affine transformation
//         if (affine_) {
//             // gamma_ and beta_ are (num_features_). Reshape for broadcasting.
//             std::vector<int64_t> affine_param_view_shape(x.dim(), 1);
//             affine_param_view_shape[1] = num_features_;
//             return x_normalized * gamma_.view(affine_param_view_shape) + beta_.view(affine_param_view_shape);
//         } else {
//             return x_normalized;
//         }
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "VirtualBatchNorm(num_features=" << num_features_
//                << ", eps=" << eps_ << ", affine=" << (affine_ ? "true" : "false")
//                << ", initialized=" << (initialized_flag_.defined() && initialized_flag_.item<bool>() ? "true" : "false") << ")";
//     }
// };
//
// // Define the public forward and init methods for the ModuleHolder
// torch::Tensor VirtualBatchNorm::forward(torch::Tensor x) {
//     return impl_->forward_impl(x);
// }
//
// void VirtualBatchNorm::init_reference_stats(const torch::Tensor& x_ref) {
//     impl_->init_reference_stats_impl(x_ref);
// }
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 3;
//     int64_t N = 2, H = 4, W = 4; // For general inputs
//     int64_t N_ref = 10;          // Size of reference batch
//
//     // --- Test Case 1: VBN basic flow ---
//     std::cout << "--- Test Case 1: VBN basic flow ---" << std::endl;
//     VirtualBatchNorm vbn_module1(num_features);
//     // std::cout << vbn_module1 << std::endl;
//
//     // 1. Create a reference batch and initialize stats
//     torch::Tensor x_reference_batch = torch::randn({N_ref, num_features, H, W}) * 2.0 + 5.0; // Ref batch
//     std::cout << "Initializing reference stats with x_ref shape: " << x_reference_batch.sizes() << std::endl;
//     vbn_module1->init_reference_stats(x_reference_batch);
//     // std::cout << vbn_module1 << std::endl; // Print again to see initialized=true
//     std::cout << "Reference mu_ref (first channel): " << vbn_module1->impl_->mu_ref_.select(1,0).item<double>() << std::endl;
//     std::cout << "Reference var_ref (first channel): " << vbn_module1->impl_->var_ref_.select(1,0).item<double>() << std::endl;
//
//
//     // 2. Create a new input batch for forward pass
//     torch::Tensor x_input1 = torch::randn({N, num_features, H, W});
//     std::cout << "\nInput x1 shape for forward: " << x_input1.sizes() << std::endl;
//
//     // Forward pass (acts like eval mode of BN, using fixed stats)
//     vbn_module1->eval(); // Or train(), behavior is the same for VBN's normalization core
//     torch::Tensor y1 = vbn_module1->forward(x_input1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//     std::cout << "Output y1 mean (channel 0): " << y1.select(1,0).mean().item<double>() << std::endl;
//     std::cout << "Output y1 std (channel 0): " << y1.select(1,0).std(false).item<double>() << std::endl;
//     // The mean/std of y1 will depend on how x_input1 relates to mu_ref/var_ref.
//     // If x_input1 had same stats as x_ref, then y1 (before affine) would have mean ~0, std ~1.
//     TORCH_CHECK(y1.sizes() == x_input1.sizes(), "Output y1 shape mismatch!");
//
//
//     // --- Test Case 2: Without affine transform ---
//     std::cout << "\n--- Test Case 2: VBN without affine ---" << std::endl;
//     VirtualBatchNorm vbn_module2(num_features, 1e-5, /*affine=*/false);
//     vbn_module2->init_reference_stats(x_reference_batch); // Re-use ref batch
//
//     torch::Tensor x_input2 = x_reference_batch.slice(0, 0, N); // Use first N samples of ref_batch as input
//                                                               // So, input stats match reference stats.
//     std::cout << "Input x2 (subset of x_ref) shape: " << x_input2.sizes() << std::endl;
//     torch::Tensor y2 = vbn_module2->forward(x_input2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     // Since x_input2 has same stats as x_reference_batch and no affine,
//     // output y2 should have mean ~0 and std ~1 for each channel.
//     std::cout << "Output y2 mean (channel 0, should be ~0): " << y2.select(1,0).mean().item<double>() << std::endl;
//     std::cout << "Output y2 std (channel 0, should be ~1): " << y2.select(1,0).std(false).item<double>() << std::endl;
//     TORCH_CHECK(std::abs(y2.select(1,0).mean().item<double>()) < 1e-1, "Mean for y2 (no affine, matched stats) not ~0."); // Looser tolerance
//     TORCH_CHECK(std::abs(y2.select(1,0).std(false).item<double>() - 1.0) < 1e-1, "Std for y2 (no affine, matched stats) not ~1.");
//
//
//     // --- Test Case 3: Check backward pass ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     VirtualBatchNorm vbn_module3(num_features, 1e-5, /*affine=*/true);
//     vbn_module3->init_reference_stats(x_reference_batch);
//     vbn_module3->train();
//
//     torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
//     torch::Tensor y3 = vbn_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_gamma = vbn_module3->impl_->gamma_.grad().defined() &&
//                              vbn_module3->impl_->gamma_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_beta = vbn_module3->impl_->beta_.grad().defined() &&
//                             vbn_module3->impl_->beta_.grad().abs().sum().item<double>() > 0;
//     // Reference stats are buffers and should not have gradients
//     bool no_grad_mu_ref = !vbn_module3->impl_->mu_ref_.grad_fn().defined();
//
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for gamma: " << (grad_exists_gamma ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for beta: " << (grad_exists_beta ? "true" : "false") << std::endl;
//     std::cout << "No gradient for mu_ref (buffer): " << (no_grad_mu_ref ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_gamma, "No gradient for gamma!");
//     TORCH_CHECK(grad_exists_beta, "No gradient for beta!");
//     TORCH_CHECK(no_grad_mu_ref, "mu_ref buffer should not have gradient!");
//
//     // Test uninitialized forward
//     std::cout << "\n--- Test Case 4: Forward before init (should fail) ---" << std::endl;
//     VirtualBatchNorm vbn_module4(num_features);
//     torch::Tensor x4 = torch::randn({N, num_features, H, W});
//     try {
//         vbn_module4->forward(x4);
//     } catch (const c10::Error& e) {
//         std::cout << "Caught expected error: " << e.what() << std::endl;
//     }
//
//
//     std::cout << "\nVirtualBatchNorm tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    VirtualBatchNorm::VirtualBatchNorm(int64_t num_features, double eps, bool affine)
        : num_features_(num_features),
          eps_(eps),
          affine_(affine)
    {
        TORCH_CHECK(num_features > 0, "num_features must be positive.");

        if (affine_)
        {
            gamma_ = register_parameter("weight", torch::ones({num_features_}));
            beta_ = register_parameter("bias", torch::zeros({num_features_}));
        }

        // Initialize reference stat buffers as undefined or placeholders.
        // They will be properly sized and filled by init_reference_stats.
        // Registering them ensures they are part of the module's state.
        mu_ref_ = register_buffer("mu_ref", torch::Tensor());
        var_ref_ = register_buffer("var_ref", torch::Tensor());
        initialized_flag_ = register_buffer("initialized_flag", torch::tensor(0, torch::kBool));
    }

    auto VirtualBatchNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);


        TORCH_CHECK(initialized_flag_.item<bool>(),
                    "VirtualBatchNorm reference statistics not initialized. Call init_reference_stats() first.");
        TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));

        // mu_ref_ and var_ref_ are stored, e.g., as (1, C, 1, 1) if initialized with 4D x_ref.
        // They need to broadcast with x.
        // If mu_ref_/var_ref_ are (C,) and x is (N,C,H,W), then reshape to (1,C,1,1).
        // Our init_reference_stats stores them in broadcastable form (e.g., 1,C,1,1 for 4D x_ref).
        // If x.dim() is different from the dim assumed during init, this might need adjustment.
        // For this implementation, assume mu_ref_ and var_ref_ are already in a shape
        // like (1, C, 1, ..., 1) that matches x.dim() after channel dim.
        TORCH_CHECK(mu_ref_.dim() == x.dim() && var_ref_.dim() == x.dim(),
                    "Dimensionality of stored reference stats (mu_ref dim: ", mu_ref_.dim(),
                    ") must match input x dim (", x.dim(), ") for broadcasting. "
                    "Ensure init_reference_stats was called with x_ref of appropriate dimensionality "
                    "or adjust reshaping. Current mu_ref shape: ", mu_ref_.sizes());


        // Normalize x using the stored reference statistics mu_ref_ and var_ref_
        torch::Tensor x_normalized = (x - mu_ref_) / torch::sqrt(var_ref_ + eps_);

        // Apply learnable affine transformation
        if (affine_)
        {
            // gamma_ and beta_ are (num_features_). Reshape for broadcasting.
            std::vector<int64_t> affine_param_view_shape(x.dim(), 1);
            affine_param_view_shape[1] = num_features_;
            return x_normalized * gamma_.view(affine_param_view_shape) + beta_.view(affine_param_view_shape);
        }
        else
        {
            return x_normalized;
        }
    }
}
