#include <normalizations/weight_standardization.h>


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <cmath> // For std::sqrt
//
// // Forward declaration for the Impl struct
// struct WsConv2dImpl;
//
// // The main module struct that users will interact with.
// // This module acts like nn::Conv2d but with Weight Standardization.
// struct WsConv2d : torch::nn::ModuleHolder<WsConv2dImpl> {
//     WsConv2d(int64_t in_channels, int64_t out_channels,
//              torch::ExpandingArray<2> kernel_size,
//              torch::ExpandingArray<2> stride = 1,
//              torch::ExpandingArray<2> padding = 0,
//              torch::ExpandingArray<2> dilation = 1,
//              int64_t groups = 1,
//              bool bias = true,
//              double ws_eps = 1e-5); // Epsilon for weight standardization
//
//     torch::Tensor forward(torch::Tensor x);
// };
//
// // The implementation struct for WsConv2d
// struct WsConv2dImpl : torch::nn::Module {
//     int64_t in_channels_;
//     int64_t out_channels_;
//     std::vector<int64_t> kernel_size_vec_; // Stored as std::vector
//     std::vector<int64_t> stride_vec_;
//     std::vector<int64_t> padding_vec_;
//     std::vector<int64_t> dilation_vec_;
//     int64_t groups_;
//     bool bias_defined_;
//     double ws_eps_; // Epsilon for Weight Standardization
//
//     // Learnable parameters for the convolution
//     // The 'weight_' parameter here is the one that gets standardized.
//     torch::Tensor weight_; // Convolution weights: (out_channels, in_channels / groups, kH, kW)
//     torch::Tensor bias_;   // Optional convolution bias
//
// public:
//     WsConv2dImpl(int64_t in_channels, int64_t out_channels,
//                  torch::ExpandingArray<2> kernel_size,
//                  torch::ExpandingArray<2> stride,
//                  torch::ExpandingArray<2> padding,
//                  torch::ExpandingArray<2> dilation,
//                  int64_t groups,
//                  bool bias,
//                  double ws_eps)
//         : in_channels_(in_channels),
//           out_channels_(out_channels),
//           kernel_size_vec_({kernel_size->at(0), kernel_size->at(1)}),
//           stride_vec_({stride->at(0), stride->at(1)}),
//           padding_vec_({padding->at(0), padding->at(1)}),
//           dilation_vec_({dilation->at(0), dilation->at(1)}),
//           groups_(groups),
//           bias_defined_(bias),
//           ws_eps_(ws_eps) {
//
//         TORCH_CHECK(in_channels_ > 0, "in_channels must be positive.");
//         TORCH_CHECK(out_channels_ > 0, "out_channels must be positive.");
//         TORCH_CHECK(kernel_size_vec_[0] > 0 && kernel_size_vec_[1] > 0, "kernel_size dimensions must be positive.");
//         TORCH_CHECK(groups_ > 0, "groups must be positive.");
//         TORCH_CHECK(in_channels_ % groups_ == 0, "in_channels must be divisible by groups.");
//         TORCH_CHECK(out_channels_ % groups_ == 0, "out_channels must be divisible by groups.");
//
//         // Initialize convolution weights
//         weight_ = register_parameter("weight", torch::randn({out_channels_, in_channels_ / groups_, kernel_size_vec_[0], kernel_size_vec_[1]}));
//         // Standard Conv2d initialization (e.g., kaiming_uniform_)
//         torch::nn::init::kaiming_uniform_(weight_, std::sqrt(5));
//
//         if (bias_defined_) {
//             bias_ = register_parameter("bias", torch::zeros({out_channels_}));
//             // Match Conv2d bias init
//             double fan_in = 1.0;
//             for(size_t i=1; i<weight_.sizes().size(); ++i) fan_in *= weight_.size(i); // C_in * kH * kW (approx)
//             double bound = 1.0 / std::sqrt(fan_in);
//              if (bias_.requires_grad()) // check if bias is a parameter that requires grad
//                 torch::nn::init::uniform_(bias_, -bound, bound);
//         }
//     }
//
//     // Computes the standardized weight W_hat
//     torch::Tensor compute_standardized_weight() {
//         // weight_ shape: (C_out, C_in_g, kH, kW) where C_in_g = C_in / groups
//         // We need to standardize each filter W[j, :, :, :] (i.e., for each C_out).
//         // The mean and std are computed over C_in_g, kH, kW dimensions for each output filter.
//
//         // Keep C_out dimension, reduce over others (1, 2, 3 for C_in_g, kH, kW)
//         std::vector<int64_t> reduce_dims = {1, 2, 3};
//
//         // Calculate mean per output filter
//         // keepdim=true to maintain shape for broadcasting (e.g., (C_out, 1, 1, 1))
//         torch::Tensor mean = weight_.mean(reduce_dims, /*keepdim=*/true);
//
//         // Calculate variance per output filter: Var(W_j) = E[W_j^2] - (E[W_j])^2
//         // Or directly use std: std(W_j)
//         // Using unbiased=false for population std, as is common in these contexts.
//         torch::Tensor std = weight_.std(reduce_dims, /*unbiased=*/false, /*keepdim=*/true);
//
//         // Standardize weights: (W - mean) / (std + eps)
//         torch::Tensor weight_hat = (weight_ - mean) / (std + ws_eps_);
//
//         return weight_hat;
//     }
//
//     torch::Tensor forward_impl(torch::Tensor x) {
//         // x: input tensor of shape (N, C_in, H_in, W_in)
//
//         TORCH_CHECK(x.dim() == 4, "Input tensor x must be 4D. Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == in_channels_,
//                     "Input x channels (", x.size(1), ") must match in_channels (", in_channels_, ").");
//
//         // Get the standardized weights
//         torch::Tensor weight_standardized = compute_standardized_weight();
//
//         // Perform convolution using the standardized weights
//         return torch::conv2d(x, weight_standardized, bias_,
//                              torch::IntArrayRef(stride_vec_),
//                              torch::IntArrayRef(padding_vec_),
//                              torch::IntArrayRef(dilation_vec_),
//                              groups_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "WsConv2d(in_channels=" << in_channels_
//                << ", out_channels=" << out_channels_
//                << ", kernel_size=[" << kernel_size_vec_[0] << "," << kernel_size_vec_[1] << "]"
//                << ", stride=[" << stride_vec_[0] << "," << stride_vec_[1] << "]"
//                << ", padding=[" << padding_vec_[0] << "," << padding_vec_[1] << "]"
//                << ", groups=" << groups_
//                << ", bias=" << (bias_defined_ ? "true" : "false")
//                << ", ws_eps=" << ws_eps_ << ")";
//     }
// };
//
//
// // Define public methods for the ModuleHolder
// WsConv2d::WsConv2d(int64_t in_channels, int64_t out_channels,
//                    torch::ExpandingArray<2> kernel_size,
//                    torch::ExpandingArray<2> stride,
//                    torch::ExpandingArray<2> padding,
//                    torch::ExpandingArray<2> dilation,
//                    int64_t groups,
//                    bool bias,
//                    double ws_eps)
//     : ModuleHolder(std::make_shared<WsConv2dImpl>(in_channels, out_channels, kernel_size,
//                                                  stride, padding, dilation, groups, bias, ws_eps)) {}
//
// torch::Tensor WsConv2d::forward(torch::Tensor x) {
//     return impl_->forward_impl(x);
// }
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t N = 2;
//     int64_t C_in = 3;
//     int64_t C_out = 5;
//     int64_t H_in = 8, W_in = 8;
//     std::vector<long> kernel_s = {3,3};
//     std::vector<long> padding_s = {1,1};
//
//     // --- Test Case 1: WsConv2d basic functionality ---
//     std::cout << "--- Test Case 1: WsConv2d defaults ---" << std::endl;
//     WsConv2d ws_conv1(C_in, C_out, kernel_s, 1, padding_s);
//     // std::cout << ws_conv1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, C_in, H_in, W_in});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//
//     // Check original weights
//     std::cout << "Original weight_ mean (filter 0): " << ws_conv1->impl_->weight_[0].mean().item<double>() << std::endl;
//     std::cout << "Original weight_ std (filter 0): " << ws_conv1->impl_->weight_[0].std(false).item<double>() << std::endl;
//
//
//     torch::Tensor y1 = ws_conv1->forward(x1);
//     std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
//     TORCH_CHECK(y1.size(0) == N && y1.size(1) == C_out && y1.size(2) == H_in && y1.size(3) == W_in,
//                 "Output y1 shape mismatch!");
//
//     // Check standardized weights used in forward (conceptually)
//     torch::Tensor standardized_w = ws_conv1->impl_->compute_standardized_weight();
//     std::cout << "Standardized weight_hat mean (filter 0, should be ~0): " << standardized_w[0].mean().item<double>() << std::endl;
//     std::cout << "Standardized weight_hat std (filter 0, should be ~1): " << standardized_w[0].std(false).item<double>() << std::endl;
//     TORCH_CHECK(std::abs(standardized_w[0].mean().item<double>()) < 1e-6, "Standardized weight mean not ~0.");
//     TORCH_CHECK(std::abs(standardized_w[0].std(false).item<double>() - 1.0) < 1e-6, "Standardized weight std not ~1.");
//
//
//     // --- Test Case 2: WsConv2d with groups ---
//     // WS applies per output filter, so groups should still work.
//     // For WS + groups, each output filter is still normalized independently over its C_in_g * kH * kW elements.
//     std::cout << "\n--- Test Case 2: WsConv2d with groups=C_in (Depthwise-like) ---" << std::endl;
//     int64_t C_in_g = 3, C_out_g = 3; // For depthwise, C_out must be multiple of C_in. Here C_out = C_in.
//     TORCH_CHECK(C_out_g % C_in_g == 0, "For depthwise, C_out must be a multiple of C_in, and groups = C_in.");
//     WsConv2d ws_conv2_grouped(C_in_g, C_out_g, kernel_s, 1, padding_s, 1, /*groups=*/C_in_g, false); // No bias
//     // std::cout << ws_conv2_grouped << std::endl;
//     // Weight shape for this grouped conv: (C_out_g, C_in_g / groups, kH, kW) -> (3, 3/3, 3, 3) -> (3, 1, 3, 3)
//
//     torch::Tensor x2 = torch::randn({N, C_in_g, H_in, W_in});
//     torch::Tensor y2 = ws_conv2_grouped->forward(x2);
//     std::cout << "Output y2 (grouped) shape: " << y2.sizes() << std::endl;
//     TORCH_CHECK(y2.size(0) == N && y2.size(1) == C_out_g && y2.size(2) == H_in && y2.size(3) == W_in,
//                 "Output y2 (grouped) shape mismatch!");
//     torch::Tensor standardized_w_g = ws_conv2_grouped->impl_->compute_standardized_weight();
//     std::cout << "Standardized weight_hat (grouped) mean (filter 0): " << standardized_w_g[0].mean().item<double>() << std::endl;
//     std::cout << "Standardized weight_hat (grouped) std (filter 0): " << standardized_w_g[0].std(false).item<double>() << std::endl;
//
//
//     // --- Test Case 3: Check backward pass ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     WsConv2d ws_conv3(C_in, C_out, kernel_s, 1, padding_s);
//     ws_conv3->train();
//
//     torch::Tensor x3 = torch::randn({N, C_in, H_in, W_in}, torch::requires_grad());
//     torch::Tensor y3 = ws_conv3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     // Gradients are w.r.t. the original `weight_` parameter, not the standardized one.
//     // The reparameterization is part of the forward graph.
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_weight = ws_conv3->impl_->weight_.grad().defined() &&
//                               ws_conv3->impl_->weight_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_bias = ws_conv3->impl_->bias_.grad().defined() && // If bias_defined_ = true
//                             ws_conv3->impl_->bias_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for original conv weight: " << (grad_exists_weight ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for conv bias: " << (grad_exists_bias ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_weight, "No gradient for original conv weight!");
//     if (ws_conv3->impl_->bias_defined_) {
//         TORCH_CHECK(grad_exists_bias, "No gradient for conv bias!");
//     }
//
//     std::cout << "\nWsConv2d (Weight Standardization) tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    auto WeightStandardization::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
