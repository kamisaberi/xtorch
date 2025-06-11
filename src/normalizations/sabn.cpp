#include "include/normalizations/sabn.h"


// ##  Switchable Atrous Batch Normalization

// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct SabnImpl;
//
// // The main module struct that users will interact with.
// // SABN - Interpreted as Switchable Activated Batch Normalization
// // (mixing two BN branches and then activating)
// struct Sabn : torch::nn::ModuleHolder<SabnImpl> {
//     using torch::nn::ModuleHolder<SabnImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for Sabn
// struct SabnImpl : torch::nn::Module {
//     int64_t num_features_;
//     double eps_;
//     double momentum_;
//     // Affine parameters are part of each BN branch
//
//     // Activation parameters (for LeakyReLU)
//     double leaky_relu_slope_;
//
//     // Components for two Batch Normalization branches (BN1, BN2)
//     // Branch 1
//     torch::Tensor running_mean1_;
//     torch::Tensor running_var1_;
//     torch::Tensor gamma1_;
//     torch::Tensor beta1_;
//     torch::Tensor num_batches_tracked1_;
//
//     // Branch 2
//     torch::Tensor running_mean2_;
//     torch::Tensor running_var2_;
//     torch::Tensor gamma2_;
//     torch::Tensor beta2_;
//     torch::Tensor num_batches_tracked2_;
//
//     // Learnable mixing weights (for combining outputs of BN1 and BN2)
//     // We'll learn `mixing_logits_` and pass them through softmax to get weights for each branch.
//     // For two branches, we only need one set of logits per channel, representing weight for branch1.
//     // Weight for branch2 will be (1 - weight_branch1).
//     // Or, more generally, K logits for K branches, then softmax. Let's use 2 logits for 2 branches.
//     torch::Tensor mixing_logits_; // Shape (1, C, 1, 1, num_branches=2)
//
//     SabnImpl(int64_t num_features,
//              double eps = 1e-5,
//              double momentum = 0.1,
//              double leaky_relu_slope = 0.01)
//         : num_features_(num_features),
//           eps_(eps),
//           momentum_(momentum),
//           leaky_relu_slope_(leaky_relu_slope) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//
//         // --- BN Branch 1 Parameters ---
//         running_mean1_ = register_buffer("running_mean1", torch::zeros({num_features_}));
//         running_var1_ = register_buffer("running_var1", torch::ones({num_features_}));
//         gamma1_ = register_parameter("gamma1", torch::ones({num_features_}));
//         beta1_ = register_parameter("beta1", torch::zeros({num_features_}));
//         num_batches_tracked1_ = register_buffer("num_batches_tracked1", torch::tensor(0, torch::kLong));
//
//         // --- BN Branch 2 Parameters ---
//         running_mean2_ = register_buffer("running_mean2", torch::zeros({num_features_}));
//         running_var2_ = register_buffer("running_var2", torch::ones({num_features_}));
//         gamma2_ = register_parameter("gamma2", torch::ones({num_features_}));
//         beta2_ = register_parameter("beta2", torch::zeros({num_features_}));
//         num_batches_tracked2_ = register_buffer("num_batches_tracked2", torch::tensor(0, torch::kLong));
//
//         // --- Mixing Weights ---
//         // Initialize logits to be equal (e.g., zeros), so initial weights are 0.5 for each branch.
//         // Shape (1, num_features, 1, 1, 2) -> for broadcasting with (N,C,H,W) and then for softmax over last dim.
//         // Or (1, num_features * 2, 1, 1) and reshape.
//         // Let's use (channels, num_branches) for logits, then expand.
//         mixing_logits_ = register_parameter("mixing_logits", torch::zeros({num_features_, 2}));
//     }
//
//     // Helper function for a single BN branch forward pass
//     torch::Tensor bn_branch_forward(
//         const torch::Tensor& x_input,
//         torch::Tensor& running_mean,
//         torch::Tensor& running_var,
//         torch::Tensor& num_batches_tracked,
//         const torch::Tensor& gamma,
//         const torch::Tensor& beta,
//         const std::vector<int64_t>& reduce_dims_stats,
//         const std::vector<int64_t>& param_view_shape) {
//
//         torch::Tensor current_mean, current_var;
//         if (this->is_training()) {
//             current_mean = x_input.mean(reduce_dims_stats, false);
//             current_var = (x_input - current_mean.view(param_view_shape)).pow(2).mean(reduce_dims_stats, false);
//
//             running_mean = (1.0 - momentum_) * running_mean + momentum_ * current_mean.detach();
//             running_var  = (1.0 - momentum_) * running_var  + momentum_ * current_var.detach();
//             num_batches_tracked += 1;
//         } else {
//             current_mean = running_mean;
//             current_var = running_var;
//         }
//
//         torch::Tensor x_bn = (x_input - current_mean.view(param_view_shape)) /
//                              torch::sqrt(current_var.view(param_view_shape) + eps_);
//         return x_bn * gamma.view(param_view_shape) + beta.view(param_view_shape);
//     }
//
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x expected to be (N, C, D1, D2, ...) e.g., 4D (N,C,H,W)
//         TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
//
//         // Prepare shapes for BN params and reduction dims
//         std::vector<int64_t> reduce_dims_stats; // Dims for mean/var (0, 2, 3, ...)
//         reduce_dims_stats.push_back(0); // Batch dimension
//         for (int64_t i = 2; i < x.dim(); ++i) {
//             reduce_dims_stats.push_back(i);
//         }
//         std::vector<int64_t> param_view_shape(x.dim(), 1); // (1,C,1,1 for NCHW)
//         param_view_shape[1] = num_features_;
//
//
//         // --- Calculate outputs of the two BN branches ---
//         torch::Tensor y_bn1 = bn_branch_forward(x, running_mean1_, running_var1_, num_batches_tracked1_,
//                                                 gamma1_, beta1_, reduce_dims_stats, param_view_shape);
//         torch::Tensor y_bn2 = bn_branch_forward(x, running_mean2_, running_var2_, num_batches_tracked2_,
//                                                 gamma2_, beta2_, reduce_dims_stats, param_view_shape);
//
//         // --- Calculate mixing weights ---
//         // mixing_logits_ is (C, 2). We want weights (1, C, 1, 1, 2) then reduce.
//         // Or (N, C, H, W, 2) effectively.
//         // Let's make weights (C, 2) -> softmax -> (C, 2)
//         // Then select w1 as (C,1) and w2 as (C,1)
//         // Then reshape to (1,C,1,1) for broadcasting
//         torch::Tensor weights = torch::softmax(mixing_logits_, /*dim=*/1); // Softmax over the "branches" dim (dim 1)
//                                                                        // weights shape: (num_features, 2)
//
//         torch::Tensor w1 = weights.select(1, 0).view(param_view_shape); // Weight for branch 1, reshaped to (1,C,1,1,..)
//         torch::Tensor w2 = weights.select(1, 1).view(param_view_shape); // Weight for branch 2, reshaped
//
//         // --- Mix the outputs ---
//         torch::Tensor mixed_bn_output = w1 * y_bn1 + w2 * y_bn2;
//
//         // --- Apply Activation (LeakyReLU) ---
//         torch::Tensor output = torch::leaky_relu(mixed_bn_output, leaky_relu_slope_);
//
//         return output;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "Sabn(num_features=" << num_features_
//                << ", eps=" << eps_ << ", momentum=" << momentum_
//                << ", leaky_relu_slope=" << leaky_relu_slope_
//                << ", num_branches=2)";
//     }
// };
// TORCH_MODULE(Sabn);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 32;
//     int64_t N = 4, H = 16, W = 16;
//
//     // --- Test Case 1: SABN with defaults ---
//     std::cout << "--- Test Case 1: SABN defaults ---" << std::endl;
//     Sabn sabn_module1(num_features);
//     // std::cout << sabn_module1 << std::endl;
//     std::cout << "Initial mixing_logits (all zeros): \n" << sabn_module1->mixing_logits_.slice(0,0,2) << std::endl;
//     std::cout << "Initial mixing weights (softmax(zeros) = 0.5 each): \n"
//               << torch::softmax(sabn_module1->mixing_logits_, 1).slice(0,0,2) << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_features, H, W});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//
//     // Training pass
//     sabn_module1->train();
//     torch::Tensor y1_train = sabn_module1->forward(x1);
//     std::cout << "Output y1_train shape: " << y1_train.sizes() << std::endl;
//     std::cout << "y1_train mean (all): " << y1_train.mean().item<double>() << std::endl;
//
//     // Evaluation pass
//     sabn_module1->eval();
//     torch::Tensor y1_eval = sabn_module1->forward(x1); // Should use running stats for both BNs
//     std::cout << "Output y1_eval shape: " << y1_eval.sizes() << std::endl;
//     std::cout << "y1_eval mean (all): " << y1_eval.mean().item<double>() << std::endl;
//     TORCH_CHECK(!torch::allclose(y1_train.mean(), y1_eval.mean()),
//                 "Train and Eval output means should differ due to BN parts.");
//
//
//     // --- Test Case 2: Forcing mixing weights ---
//     std::cout << "\n--- Test Case 2: Forcing mixing weights ---" << std::endl;
//     Sabn sabn_module2(num_features);
//     // Force weights to favor branch1 (e.g., logits [10, 0] -> softmax ~[1, 0])
//     sabn_module2->mixing_logits_.data().slice(1,0,1).fill_(10.0); // logits for branch1
//     sabn_module2->mixing_logits_.data().slice(1,1,2).fill_(0.0);  // logits for branch2
//     std::cout << "Forced mixing_logits (example for first feature): " << sabn_module2->mixing_logits_[0] << std::endl;
//     std::cout << "Resulting weights (example for first feature): "
//               << torch::softmax(sabn_module2->mixing_logits_[0], 0) << std::endl;
//
//     // Modify affine params of branch1 to make it distinct
//     sabn_module2->gamma1_.data().fill_(2.0);
//     sabn_module2->beta1_.data().fill_(1.0);
//     // Keep branch2 affine params at default 1.0 and 0.0
//
//     sabn_module2->eval(); // Use running stats (initially 0 mean, 1 var for both)
//     torch::Tensor x2 = torch::randn({N, num_features, H, W});
//     torch::Tensor y2 = sabn_module2->forward(x2);
//
//     // Output y2 should mostly reflect branch1's characteristics (scaled by ~2, shifted by ~1 before activation)
//     // because its initial running stats are 0/1.
//     // The normalized x (mean 0, std 1) times gamma1 (2) plus beta1 (1) = mean 1, std 2.
//     // Then LeakyReLU.
//     std::cout << "Output y2 (favored branch1) mean (channel 0): " << y2.select(1,0).mean().item<double>() << std::endl;
//     std::cout << "Output y2 (favored branch1) std (channel 0): " << y2.select(1,0).std(false).item<double>() << std::endl;
//
//
//     // --- Test Case 3: Check backward pass ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     Sabn sabn_module3(num_features);
//     sabn_module3->train();
//
//     torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
//     torch::Tensor y3 = sabn_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_mixing_logits = sabn_module3->mixing_logits_.grad().defined() &&
//                                      sabn_module3->mixing_logits_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_gamma1 = sabn_module3->gamma1_.grad().defined() &&
//                               sabn_module3->gamma1_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_gamma2 = sabn_module3->gamma2_.grad().defined() &&
//                               sabn_module3->gamma2_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for mixing_logits: " << (grad_exists_mixing_logits ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for gamma1: " << (grad_exists_gamma1 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for gamma2: " << (grad_exists_gamma2 ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_mixing_logits, "No gradient for mixing_logits!");
//     TORCH_CHECK(grad_exists_gamma1, "No gradient for gamma1!");
//     TORCH_CHECK(grad_exists_gamma2, "No gradient for gamma2!");
//
//     std::cout << "\nSabn (Switchable Activated Batch Normalization) tests finished." << std::endl;
//     return 0;
// }





namespace xt::norm
{
    auto SABN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
