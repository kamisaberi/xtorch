#include "include/normalizations/mpn.h"

//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct MpnNormalizationImpl;
//
// // The main module struct that users will interact with.
// // "MPN" - Interpreted here as Message Passing Normalization,
// // implemented using Layer Normalization on node features.
// struct MpnNormalization : torch::nn::ModuleHolder<MpnNormalizationImpl> {
//     using torch::nn::ModuleHolder<MpnNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for MpnNormalization
// struct MpnNormalizationImpl : torch::nn::Module {
//     // This implementation will use Layer Normalization.
//     // The 'normalized_shape' for LayerNorm will be the feature dimension.
//     std::vector<int64_t> normalized_shape_; // Should be {num_features}
//     double eps_;
//     bool elementwise_affine_; // Whether LayerNorm has learnable gamma and beta
//
//     // Layer Normalization components
//     torch::Tensor gamma_; // Scale (if elementwise_affine is true)
//     torch::Tensor beta_;  // Shift (if elementwise_affine is true)
//
//     MpnNormalizationImpl(const std::vector<int64_t>& normalized_shape, // e.g., {num_features}
//                          double eps = 1e-5,
//                          bool elementwise_affine = true)
//         : normalized_shape_(normalized_shape),
//           eps_(eps),
//           elementwise_affine_(elementwise_affine) {
//
//         TORCH_CHECK(!normalized_shape_.empty(), "normalized_shape cannot be empty.");
//         // For typical node features (NumNodes, NumFeatures) or (N, MaxNodes, NumFeatures),
//         // normalized_shape_ would be {NumFeatures}.
//
//         if (elementwise_affine_) {
//             // Gamma and beta will have the same shape as normalized_shape_
//             gamma_ = register_parameter("weight", torch::ones(normalized_shape_)); // PyTorch LN naming
//             beta_  = register_parameter("bias",   torch::zeros(normalized_shape_));  // PyTorch LN naming
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x: (..., F1, F2, ..., Fk) where {F1, ..., Fk} matches normalized_shape_
//         // For GNNs, common shapes:
//         // - (NumTotalNodes, NumFeatures) -> normalized_shape_ = {NumFeatures}
//         // - (BatchSize, NumNodesPerGraph, NumFeatures) -> normalized_shape_ = {NumFeatures}
//
//         // Check that the trailing dimensions of x match normalized_shape_
//         TORCH_CHECK(x.dim() >= normalized_shape_.size(),
//                     "Input tensor rank (", x.dim(), ") must be at least the rank of normalized_shape (", normalized_shape_.size(), ").");
//         for (size_t i = 0; i < normalized_shape_.size(); ++i) {
//             TORCH_CHECK(x.size(-normalized_shape_.size() + i) == normalized_shape_[i],
//                         "Trailing dimensions of input x do not match normalized_shape. Expected shape ending with ",
//                         torch::IntArrayRef(normalized_shape_), ", got input shape ", x.sizes());
//         }
//
//         // --- Layer Normalization Logic ---
//         // Mean and variance are computed over the 'normalized_shape_' dimensions.
//         // These are the last D dimensions, where D = normalized_shape_.size().
//         // For example, if normalized_shape_ = {C}, we normalize over the last dimension.
//         // If normalized_shape_ = {H, W}, we normalize over the last two dimensions.
//
//         std::vector<int64_t> reduction_axes;
//         for (size_t i = 0; i < normalized_shape_.size(); ++i) {
//             reduction_axes.push_back(-(static_cast<int64_t>(normalized_shape_.size()) - i)); // -D, -D+1, ..., -1
//         }
//
//         // keepdim=true to allow broadcasting
//         auto mean = x.mean(reduction_axes, /*keepdim=*/true);
//         // unbiased=false for population variance
//         auto var = x.var(reduction_axes, /*unbiased=*/false, /*keepdim=*/true);
//
//         torch::Tensor x_normalized = (x - mean) / torch::sqrt(var + eps_);
//
//         if (elementwise_affine_) {
//             // gamma_ and beta_ have shape normalized_shape_.
//             // They need to be broadcastable with x_normalized.
//             // If normalized_shape_ is {C} and x is (N,L,C), gamma_ is (C).
//             // It will broadcast correctly during multiplication.
//             return x_normalized * gamma_ + beta_;
//         } else {
//             return x_normalized;
//         }
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "MpnNormalization(normalized_shape=[";
//         for (size_t i = 0; i < normalized_shape_.size(); ++i) {
//             stream << normalized_shape_[i] << (i == normalized_shape_.size() - 1 ? "" : ",");
//         }
//         stream << "], eps=" << eps_
//                << ", elementwise_affine=" << (elementwise_affine_ ? "true" : "false") << ")";
//     }
// };
// TORCH_MODULE(MpnNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     // --- Test Case 1: Input like (NumNodes, NumFeatures) ---
//     std::cout << "--- Test Case 1: Input (NumNodes, NumFeatures) ---" << std::endl;
//     int64_t num_nodes1 = 10;
//     int64_t num_features1 = 32;
//     MpnNormalization mpn_module1({num_features1}); // normalized_shape is {num_features1}
//     // std::cout << mpn_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({num_nodes1, num_features1});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//
//     torch::Tensor y1 = mpn_module1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//
//     // Check mean and std of one node's features after normalization
//     // (should be ~0 and ~1 if affine params are default 1 and 0)
//     std::cout << "y1[0] mean (should be ~0): " << y1[0].mean().item<double>() << std::endl;
//     std::cout << "y1[0] std (should be ~1):  " << y1[0].std(false).item<double>() << std::endl; // Population std
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//     TORCH_CHECK(std::abs(y1[0].mean().item<double>()) < 1e-6, "Mean of y1[0] not close to 0.");
//     TORCH_CHECK(std::abs(y1[0].std(false).item<double>() - 1.0) < 1e-6, "Std of y1[0] not close to 1.");
//
//
//     // --- Test Case 2: Input like (BatchSize, NumNodesPerGraph, NumFeatures) ---
//     std::cout << "\n--- Test Case 2: Input (Batch, Nodes, Features) ---" << std::endl;
//     int64_t batch_size2 = 4;
//     int64_t num_nodes_pg2 = 7;
//     int64_t num_features2 = 16;
//     MpnNormalization mpn_module2({num_features2}); // normalized_shape is {num_features2}
//     // std::cout << mpn_module2 << std::endl;
//     // Modify affine params to see their effect
//     mpn_module2->gamma_.data().fill_(1.5);
//     mpn_module2->beta_.data().fill_(0.5);
//
//
//     torch::Tensor x2 = torch::randn({batch_size2, num_nodes_pg2, num_features2});
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//
//     torch::Tensor y2 = mpn_module2->forward(x2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//
//     // Check mean and std of features for one node in one batch item
//     // (should be ~beta (0.5) and ~gamma (1.5) respectively)
//     auto one_node_features_y2 = y2.select(0,0).select(0,0); // y2[0,0,:]
//     std::cout << "y2[0,0,:] mean (should be ~0.5): " << one_node_features_y2.mean().item<double>() << std::endl;
//     std::cout << "y2[0,0,:] std (should be ~1.5):  " << one_node_features_y2.std(false).item<double>() << std::endl;
//     TORCH_CHECK(y2.sizes() == x2.sizes(), "Output y2 shape mismatch!");
//     TORCH_CHECK(std::abs(one_node_features_y2.mean().item<double>() - 0.5) < 1e-6, "Mean of y2[0,0,:] not close to 0.5.");
//     TORCH_CHECK(std::abs(one_node_features_y2.std(false).item<double>() - 1.5) < 1e-6, "Std of y2[0,0,:] not close to 1.5.");
//
//
//     // --- Test Case 3: Higher rank normalized_shape (e.g., for image-like features per node) ---
//     // This is less common for "node features" in GNNs but LayerNorm supports it.
//     std::cout << "\n--- Test Case 3: Input (Nodes, C, H, W), normalized_shape={C,H,W} ---" << std::endl;
//     int64_t num_nodes3 = 5;
//     int64_t C3=3, H3=4, W3=4;
//     MpnNormalization mpn_module3({C3, H3, W3});
//     // std::cout << mpn_module3 << std::endl;
//
//     torch::Tensor x3 = torch::randn({num_nodes3, C3, H3, W3});
//     std::cout << "Input x3 shape: " << x3.sizes() << std::endl;
//     torch::Tensor y3 = mpn_module3->forward(x3);
//     std::cout << "Output y3 shape: " << y3.sizes() << std::endl;
//     // For y3[0,:,:,:], mean should be ~0 and std ~1 over all C*H*W elements.
//     std::cout << "y3[0] mean (should be ~0): " << y3[0].mean().item<double>() << std::endl;
//     std::cout << "y3[0] std (should be ~1):  " << y3[0].std(false).item<double>() << std::endl;
//     TORCH_CHECK(std::abs(y3[0].mean().item<double>()) < 1e-5, "Mean of y3[0] not close to 0."); // Looser tolerance due to more elements
//     TORCH_CHECK(std::abs(y3[0].std(false).item<double>() - 1.0) < 1e-5, "Std of y3[0] not close to 1.");
//
//
//     // --- Test Case 4: Check backward pass ---
//     std::cout << "\n--- Test Case 4: Backward pass check ---" << std::endl;
//     MpnNormalization mpn_module4({num_features1});
//     mpn_module4->train(); // Ensure gamma/beta have requires_grad=true
//
//     torch::Tensor x4 = torch::randn({num_nodes1, num_features1}, torch::requires_grad());
//     torch::Tensor y4 = mpn_module4->forward(x4);
//     torch::Tensor loss = y4.mean();
//     loss.backward();
//
//     bool grad_exists_x4 = x4.grad().defined() && x4.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_gamma = mpn_module4->gamma_.grad().defined() &&
//                              mpn_module4->gamma_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_beta = mpn_module4->beta_.grad().defined() &&
//                             mpn_module4->beta_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x4: " << (grad_exists_x4 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for gamma: " << (grad_exists_gamma ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for beta: " << (grad_exists_beta ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x4, "No gradient for x4!");
//     TORCH_CHECK(grad_exists_gamma, "No gradient for gamma!");
//     TORCH_CHECK(grad_exists_beta, "No gradient for beta!");
//
//     std::cout << "\nMpnNormalization (as LayerNorm) tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    MPNNorm::MPNNorm(const std::vector<int64_t>& normalized_shape, // e.g., {num_features}
                     double eps,
                     bool elementwise_affine)
        : normalized_shape_(normalized_shape),
          eps_(eps),
          elementwise_affine_(elementwise_affine)
    {
        TORCH_CHECK(!normalized_shape_.empty(), "normalized_shape cannot be empty.");
        // For typical node features (NumNodes, NumFeatures) or (N, MaxNodes, NumFeatures),
        // normalized_shape_ would be {NumFeatures}.

        if (elementwise_affine_)
        {
            // Gamma and beta will have the same shape as normalized_shape_
            gamma_ = register_parameter("weight", torch::ones(normalized_shape_)); // PyTorch LN naming
            beta_ = register_parameter("bias", torch::zeros(normalized_shape_)); // PyTorch LN naming
        }
    }

    auto MPNNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);

        // Input x: (..., F1, F2, ..., Fk) where {F1, ..., Fk} matches normalized_shape_
        // For GNNs, common shapes:
        // - (NumTotalNodes, NumFeatures) -> normalized_shape_ = {NumFeatures}
        // - (BatchSize, NumNodesPerGraph, NumFeatures) -> normalized_shape_ = {NumFeatures}

        // Check that the trailing dimensions of x match normalized_shape_
        TORCH_CHECK(x.dim() >= normalized_shape_.size(),
                    "Input tensor rank (", x.dim(), ") must be at least the rank of normalized_shape (",
                    normalized_shape_.size(), ").");
        for (size_t i = 0; i < normalized_shape_.size(); ++i)
        {
            TORCH_CHECK(x.size(-normalized_shape_.size() + i) == normalized_shape_[i],
                        "Trailing dimensions of input x do not match normalized_shape. Expected shape ending with ",
                        torch::IntArrayRef(normalized_shape_), ", got input shape ", x.sizes());
        }

        // --- Layer Normalization Logic ---
        // Mean and variance are computed over the 'normalized_shape_' dimensions.
        // These are the last D dimensions, where D = normalized_shape_.size().
        // For example, if normalized_shape_ = {C}, we normalize over the last dimension.
        // If normalized_shape_ = {H, W}, we normalize over the last two dimensions.

        std::vector<int64_t> reduction_axes;
        for (size_t i = 0; i < normalized_shape_.size(); ++i)
        {
            reduction_axes.push_back(-(static_cast<int64_t>(normalized_shape_.size()) - i)); // -D, -D+1, ..., -1
        }

        // keepdim=true to allow broadcasting
        auto mean = x.mean(reduction_axes, /*keepdim=*/true);
        // unbiased=false for population variance
        auto var = x.var(reduction_axes, /*unbiased=*/false, /*keepdim=*/true);

        torch::Tensor x_normalized = (x - mean) / torch::sqrt(var + eps_);

        if (elementwise_affine_)
        {
            // gamma_ and beta_ have shape normalized_shape_.
            // They need to be broadcastable with x_normalized.
            // If normalized_shape_ is {C} and x is (N,L,C), gamma_ is (C).
            // It will broadcast correctly during multiplication.
            return x_normalized * gamma_ + beta_;
        }
        else
        {
            return x_normalized;
        }
    }
}
