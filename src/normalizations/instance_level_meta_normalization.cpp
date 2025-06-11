#include "include/normalizations/instance_level_meta_normalization.h"


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct InstanceLevelMetaNormImpl;
//
// // The main module struct that users will interact with.
// struct InstanceLevelMetaNorm : torch::nn::ModuleHolder<InstanceLevelMetaNormImpl> {
//     using torch::nn::ModuleHolder<InstanceLevelMetaNormImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for InstanceLevelMetaNorm
// struct InstanceLevelMetaNormImpl : torch::nn::Module {
//     int64_t num_features_;          // Number of features in input x (channels)
//     double eps_in_;                 // Epsilon for Instance Normalization
//     int64_t meta_hidden_dim_;       // Hidden dimension for the meta-network
//
//     // Meta-network layers (MLP to produce gamma and beta from instance features)
//     // It will take pooled features from x as input.
//     torch::nn::AdaptiveAvgPool2d avg_pool_{nullptr}; // For NCHW inputs
//     torch::nn::Linear fc_meta1_{nullptr};
//     torch::nn::Linear fc_meta_out_{nullptr};
//
//     InstanceLevelMetaNormImpl(int64_t num_features,
//                               int64_t meta_hidden_dim = 0, // 0 or less means direct map after pooling
//                               double eps_in = 1e-5)
//         : num_features_(num_features),
//           meta_hidden_dim_(meta_hidden_dim),
//           eps_in_(eps_in) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//
//         // Meta-network for generating gamma and beta
//         // For NCHW, after AdaptiveAvgPool2d, input to fc_meta1 will be (N, C, 1, 1) -> flatten to (N, C)
//         avg_pool_ = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
//         register_module("avg_pool_meta", avg_pool_);
//
//         if (meta_hidden_dim_ <= 0) { // Direct mapping from pooled features to gamma/beta
//             fc_meta_out_ = torch::nn::Linear(num_features_, 2 * num_features_); // Input C, output 2*C
//             register_module("fc_meta_out", fc_meta_out_);
//         } else {
//             fc_meta1_ = torch::nn::Linear(num_features_, meta_hidden_dim_);
//             fc_meta_out_ = torch::nn::Linear(meta_hidden_dim_, 2 * num_features_); // Output for gamma and beta
//             register_module("fc_meta1", fc_meta1_);
//             register_module("fc_meta_out", fc_meta_out_);
//         }
//         // No explicit gamma/beta parameters for the IN part itself if they are fully predicted.
//         // Or, one could have standard IN affine params and *additionally* predict modulations.
//         // This impl assumes gamma/beta are entirely predicted.
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: input tensor, e.g., (N, C, H, W) or (N, C, L)
//         // C must be num_features_
//
//         TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
//
//         int64_t N = x.size(0);
//         // --- 1. Standard Instance Normalization ---
//         torch::Tensor x_in; // Instance Normalized x
//
//         if (x.dim() > 2) { // Input has spatial/sequential dimensions (N, C, D1, ...)
//             std::vector<int64_t> reduce_dims_in; // Dims for mean/var (D1, D2, ...)
//             for (int64_t i = 2; i < x.dim(); ++i) {
//                 reduce_dims_in.push_back(i);
//             }
//             auto mean = x.mean(reduce_dims_in, /*keepdim=*/true);
//             auto var = x.var(reduce_dims_in, /*unbiased=*/false, /*keepdim=*/true);
//             x_in = (x - mean) / torch::sqrt(var + eps_in_);
//         } else { // Input is 2D (N, C). InstanceNorm on a single point per channel results in 0.
//             x_in = torch::zeros_like(x);
//         }
//
//         // --- 2. Generate Gamma and Beta from the "Meta Network" using x itself ---
//         torch::Tensor instance_features_for_meta;
//         if (x.dim() == 4) { // NCHW
//             instance_features_for_meta = avg_pool_->forward(x);         // (N, C, 1, 1)
//             instance_features_for_meta = instance_features_for_meta.view({N, -1}); // Flatten to (N, C)
//         } else if (x.dim() == 3) { // NCL (e.g., 1D Conv)
//             // Global average pooling over L dimension
//             instance_features_for_meta = x.mean(/*dim=*/2, /*keepdim=false*/false); // (N, C)
//         } else { // x.dim() == 2 (NC)
//             instance_features_for_meta = x; // Use x directly as features
//         }
//         // instance_features_for_meta is now (N, num_features_)
//
//         torch::Tensor meta_params = instance_features_for_meta;
//         if (fc_meta1_) {
//             meta_params = fc_meta1_->forward(meta_params);
//             meta_params = torch::relu(meta_params); // Common activation for hidden layer
//         }
//         meta_params = fc_meta_out_->forward(meta_params); // (N, 2 * num_features_)
//
//         auto chunks = torch::chunk(meta_params, 2, /*dim=*/1);
//         torch::Tensor gamma_predicted = chunks[0]; // (N, num_features_)
//         torch::Tensor beta_predicted  = chunks[1]; // (N, num_features_)
//
//         // --- 3. Reshape predicted Gamma and Beta for broadcasting ---
//         // Desired shape: (N, C, 1, 1, ...) to match x_in (N, C, D1, D2, ...)
//         std::vector<int64_t> affine_param_view_shape;
//         affine_param_view_shape.push_back(N);               // N
//         affine_param_view_shape.push_back(num_features_);   // C
//         for (int64_t i = 2; i < x.dim(); ++i) {
//             affine_param_view_shape.push_back(1);
//         }
//
//         gamma_predicted = gamma_predicted.view(affine_param_view_shape);
//         beta_predicted  = beta_predicted.view(affine_param_view_shape);
//
//         // --- 4. Apply "meta" affine transformation ---
//         return gamma_predicted * x_in + beta_predicted;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "InstanceLevelMetaNorm(num_features=" << num_features_
//                << ", meta_hidden_dim=" << (fc_meta1_ ? std::to_string(meta_hidden_dim_) : "0 (direct)")
//                << ", eps_in=" << eps_in_ << ")";
//     }
// };
// TORCH_MODULE(InstanceLevelMetaNorm);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 32;
//     int64_t N = 4;
//
//     // --- Test Case 1: 4D input x (NCHW), no hidden layer in meta-net ---
//     std::cout << "--- Test Case 1: 4D input (NCHW), no hidden meta layer ---" << std::endl;
//     int64_t H = 8, W = 8;
//     InstanceLevelMetaNorm ilmn_module1(num_features, /*meta_hidden_dim=*/0);
//     // std::cout << ilmn_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_features, H, W});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//
//     ilmn_module1->eval(); // Doesn't impact IN core logic, but good practice
//     torch::Tensor y1 = ilmn_module1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//     // Mean/Std of y1 for a given instance/channel will be approx. beta_pred/gamma_pred for that instance/channel
//     std::cout << "y1 [0,0,:,:] mean: " << y1.select(0,0).select(0,0).mean().item<double>() << std::endl;
//     std::cout << "y1 [0,0,:,:] std:  " << y1.select(0,0).select(0,0).std(false).item<double>() << std::endl;
//     TORCH_CHECK(y1.sizes() == x1.sizes(), "Output y1 shape mismatch!");
//
//
//     // --- Test Case 2: 3D input x (NCL), with hidden layer in meta-net ---
//     std::cout << "\n--- Test Case 2: 3D input (NCL), with hidden meta layer ---" << std::endl;
//     int64_t L = 16;
//     int64_t meta_hidden = 16;
//     InstanceLevelMetaNorm ilmn_module2(num_features, meta_hidden);
//     // std::cout << ilmn_module2 << std::endl;
//
//     torch::Tensor x2 = torch::randn({N, num_features, L});
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//
//     torch::Tensor y2 = ilmn_module2->forward(x2);
//     std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
//     TORCH_CHECK(y2.sizes() == x2.sizes(), "Output y2 shape mismatch!");
//
//
//     // --- Test Case 3: 2D input x (NC) ---
//     std::cout << "\n--- Test Case 3: 2D input (NC) ---" << std::endl;
//     InstanceLevelMetaNorm ilmn_module_2d(num_features, 0); // No hidden meta layer
//     torch::Tensor x_2d = torch::randn({N, num_features});
//     std::cout << "Input x_2d shape: " << x_2d.sizes() << std::endl;
//     torch::Tensor y_2d = ilmn_module_2d->forward(x_2d);
//     std::cout << "Output y_2d shape: " << y_2d.sizes() << std::endl;
//     // For 2D input, x_in is 0. So y_2d should be beta_predicted.
//     // The meta_params will be derived from x_2d itself.
//     std::cout << "y_2d[0] (example, should be predicted beta for instance 0): " << y_2d[0] << std::endl;
//
//
//     // --- Test Case 4: Check backward pass ---
//     std::cout << "\n--- Test Case 4: Backward pass check ---" << std::endl;
//     InstanceLevelMetaNorm ilmn_module3(num_features, meta_hidden);
//     ilmn_module3->train(); // Ensure parameters of meta-net have requires_grad=true
//
//     torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
//     torch::Tensor y3 = ilmn_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_fc_meta_out_weight = ilmn_module3->fc_meta_out_->weight.grad().defined() &&
//                                           ilmn_module3->fc_meta_out_->weight.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for fc_meta_out.weight: " << (grad_exists_fc_meta_out_weight ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_fc_meta_out_weight, "No gradient for fc_meta_out.weight!");
//
//     std::cout << "\nInstanceLevelMetaNorm tests finished." << std::endl;
//     return 0;
// }

namespace xt::norm
{
    auto InstanceLevelMetaNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
