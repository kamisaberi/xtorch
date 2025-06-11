#include "include/normalizations/weight_demodulation.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath> // For std::sqrt

// Forward declaration for the Impl struct
struct WeightDemodulatedConv2dImpl;

// The main module struct that users will interact with.
// This module performs a 2D convolution with weight demodulation.
struct WeightDemodulatedConv2d : torch::nn::ModuleHolder<WeightDemodulatedConv2dImpl> {
    using torch::nn::ModuleHolder<WeightDemodulatedConv2dImpl>::ModuleHolder;

    // Forward method takes the input feature map x and the style scales s_scales
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& s_scales);
};

// The implementation struct for WeightDemodulatedConv2d
struct WeightDemodulatedConv2dImpl : torch::nn::Module {
    int64_t in_channels_;
    int64_t out_channels_;
    std::vector<int64_t> kernel_size_;
    std::vector<int64_t> stride_;
    std::vector<int64_t> padding_;
    std::vector<int64_t> dilation_;
    int64_t groups_;
    bool bias_defined_; // Whether this layer has a bias term
    double eps_;      // Epsilon for demodulation stability

    // Learnable parameters for the convolution
    torch::Tensor weight_; // Convolution weights (original, before modulation/demodulation)
    torch::Tensor bias_;   // Optional convolution bias

    WeightDemodulatedConv2dImpl(int64_t in_channels, int64_t out_channels,
                                torch::ExpandingArray<2> kernel_size,
                                torch::ExpandingArray<2> stride = 1,
                                torch::ExpandingArray<2> padding = 0,
                                torch::ExpandingArray<2> dilation = 1,
                                int64_t groups = 1,
                                bool bias = true,
                                double demod_eps = 1e-8)
        : in_channels_(in_channels),
          out_channels_(out_channels),
          kernel_size_({kernel_size->at(0), kernel_size->at(1)}),
          stride_({stride->at(0), stride->at(1)}),
          padding_({padding->at(0), padding->at(1)}),
          dilation_({dilation->at(0), dilation->at(1)}),
          groups_(groups),
          bias_defined_(bias),
          eps_(demod_eps) {

        TORCH_CHECK(in_channels_ > 0, "in_channels must be positive.");
        TORCH_CHECK(out_channels_ > 0, "out_channels must be positive.");
        TORCH_CHECK(kernel_size_[0] > 0 && kernel_size_[1] > 0, "kernel_size dimensions must be positive.");
        TORCH_CHECK(groups_ > 0, "groups must be positive.");
        TORCH_CHECK(in_channels_ % groups_ == 0, "in_channels must be divisible by groups.");
        TORCH_CHECK(out_channels_ % groups_ == 0, "out_channels must be divisible by groups.");


        // Initialize convolution weights
        // Shape: (out_channels, in_channels / groups, kernel_H, kernel_W)
        weight_ = register_parameter("weight", torch::randn({out_channels_, in_channels_ / groups_, kernel_size_[0], kernel_size_[1]}));
        // StyleGAN2 often uses He initialization (kaiming_normal_)
        torch::nn::init::kaiming_normal_(weight_, 0.0, torch::kFanIn, torch::kReLU); // Example init

        if (bias_defined_) {
            bias_ = register_parameter("bias", torch::zeros({out_channels_}));
        }
    }

    torch::Tensor forward_impl(const torch::Tensor& x, const torch::Tensor& s_scales) {
        // x: input feature map (N, C_in, H_in, W_in)
        // s_scales: style scaling factors (N, C_in) - one scale per input channel for each batch item

        TORCH_CHECK(x.dim() == 4, "Input feature map x must be 4D. Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == in_channels_, "Input x channels mismatch. Expected ", in_channels_, ", got ", x.size(1));
        TORCH_CHECK(s_scales.dim() == 2, "s_scales must be 2D (N, C_in). Got shape ", s_scales.sizes());
        TORCH_CHECK(s_scales.size(0) == x.size(0), "Batch size of x and s_scales must match.");
        TORCH_CHECK(s_scales.size(1) == in_channels_, "s_scales feature dim must match in_channels.");

        int64_t N = x.size(0);

        // --- 1. Modulate weights ---
        // original weight_ shape: (C_out, C_in_per_group, kH, kW)
        // s_scales shape: (N, C_in)
        // We need to produce a weight for each batch item: (N, C_out, C_in_per_group, kH, kW)
        // Or, perform grouped convolution if N can be folded into C_out.
        // StyleGAN2 often performs grouped convolution where N becomes part of groups or C_out.

        // Reshape s_scales for broadcasting with weight_:
        // s_scales: (N, C_in) -> (N, 1, C_in_per_group * groups, 1, 1) if groups > 1, or (N, 1, C_in, 1, 1)
        // For grouped convolution, C_in is actually C_in_total.
        // And weight is (C_out_total, C_in_per_group, kH, kW)
        // s_scales applies to C_in_total.
        // We need to expand s_scales to (N, C_out, C_in_per_group, 1, 1) and apply to weight.
        // This would mean s_scales is (N, C_in_per_group) effectively, repeated for groups.
        // Simpler: modulate W for each N. Output W_mod is (N * C_out, C_in_per_group, kH, kW)

        // W_orig shape: (C_out, C_in/groups, kH, kW)
        // s_scales shape: (N, C_in)
        // Modulated weight w' = s * w. Style `s` is per input channel.
        // w' must be different for each N if s_scales is different for each N.

        // Create batch-wise modulated weights
        // target w_modulated shape for grouped conv: (N * C_out, C_in/groups, kH, kW)
        // or iterate batch and do N separate convolutions (less efficient)

        // Let current_w = weight_ (C_out, C_in/G, kH, kW)
        // s_scales (N, C_in). Reshape s_scales for broadcasting: (N, 1, C_in, 1, 1)
        // For grouped conv, C_in should be thought of as C_in_total.
        // And weight is (C_out, C_in_per_group, kH, kW).
        // s_scales has C_in_total features.
        // Each filter in weight_ (indexed by C_out, C_in_per_group) needs to be scaled.
        // The C_in_per_group corresponds to a slice of s_scales.

        // StyleGAN2 approach:
        // w_prime = weight_.unsqueeze(0); // (1, C_out, C_in/G, kH, kW)
        // s = s_scales.view({N, 1, in_channels_, 1, 1}); // (N, 1, C_in, 1, 1)
        // If groups > 1, s needs to be reshaped to align with C_in/G
        // s_for_mod = s_scales.view({N, 1, groups_, in_channels_ / groups_, 1, 1});
        // w_modulated = w_prime * s_for_mod; // This requires careful broadcasting or splitting C_in of s
        // This multiplication is `s_i * w_ijk` where `i` is input channel.

        // For each batch item N:
        //  w_n = weight_.clone() // (C_out, C_in/G, kH, kW)
        //  s_n = s_scales[n]   // (C_in)
        //  For g in groups:
        //    s_n_group_g = s_n[g*(C_in/G) : (g+1)*(C_in/G)] // (C_in/G)
        //    w_n[:, :, kH, kW] where input channels are from group g, scale by s_n_group_g
        // This is complex to vectorize directly for the modulation part.

        // Alternative: fuse N into C_out for grouped convolution
        // weight is (C_out, C_in_g, kH, kW)
        // s_scales is (N, C_in_total)
        // Create dynamic weight of shape (N, C_out, C_in_g, kH, kW)
        // This is what the original StyleGAN2 code effectively does before grouping.
        torch::Tensor w = weight_.unsqueeze(0); // (1, C_out, C_in/G, kH, kW)
        // s_scales needs to broadcast to C_in/G dimension for multiplication.
        // s_scales is (N, C_in_total). We need it as (N, 1, C_in_total, 1, 1) to scale input channels.
        // If using grouped conv, C_in_total = groups * (C_in/G).
        // The style is applied per input channel *before* grouping for the convolution.
        // So, s_scales has `in_channels_` elements.
        // We need to make s_scales broadcastable to the `C_in/G` dimension of `w`.
        // `w` is (C_out, C_in/G, kH, kW). `s_scales` is (N, C_in_total).
        // Let's modulate first, then demodulate.
        // `w_mod` will be (N, C_out, C_in/G, kH, kW)
        // `s_for_mod_view` should be (N, 1, C_in_total or C_in/G if already grouped, 1, 1)
        // Assume s_scales applies to the `in_channels_` which is C_in_total.
        // It should scale the `C_in/G` dimension of the weights.
        // This means we need to select the correct part of s_scales for each group.
        // For simplicity if groups = 1:
        // s_for_mod_view = s_scales.view({N, 1, in_channels_, 1, 1});
        // w_modulated = w * s_for_mod_view; // (N, C_out, C_in, kH, kW)

        // General grouped case:
        // We need to expand s_scales to (N, 1, in_channels_, 1, 1)
        // Then for each output filter, multiply by the relevant input channel scales.
        // This is equivalent to: w_modulated[n, o, i_g, kh, kw] = weight_[o, i_g, kh, kw] * s_scales[n, map_group_to_total_input_channel(i_g, group_idx)]
        // This is tricky. StyleGAN2 code is a good reference.
        // The `weight_` is (out_channels, in_channels_group, kh, kw)
        // `s_scales` is (N, in_channels_total)
        // For each n in N, for each o in out_channels:
        //   The filter `weight_[o, :, :, :]` operates on `in_channels_group` inputs.
        //   These `in_channels_group` inputs correspond to a specific slice of `in_channels_total`.
        //   The scaling `s_scales[n, input_channel_slice]` should be applied.

        // Let's make w (N, C_out, C_in/G, kH, kW)
        // And s_scales (N, C_in_total). Reshape s to (N, 1, C_in_total, 1, 1)
        // Then modulate:
        // This still isn't quite right for grouped conv directly.
        // The original implementation in StyleGAN2 effectively does this:
        // W_ijk' = s_i * W_ijk (s_i is for input channel i)
        // Let's prepare W for a grouped convolution:
        // W is (C_out, C_in/G, kH, kW)
        // s_scales is (N, C_in) which is C_in_total
        // We need a weight of shape (N*C_out, C_in/G, kH, kW) for grouped conv on x of shape (1, N*C_in, H, W) or similar.

        // Simpler path: assume full convolution (groups=1) for modulation logic first.
        TORCH_CHECK(groups_ == 1, "WeightDemodulation grouped convolution (groups > 1) logic is complex and not fully implemented here. Set groups=1.");
        // If groups == 1, in_channels_ == C_in/G
        torch::Tensor w_modulated = weight_.unsqueeze(0) * s_scales.view({N, 1, in_channels_, 1, 1});
        // w_modulated shape: (N, C_out, C_in, kH, kW)

        // --- 2. Demodulate weights ---
        // w''_ijk = w'_ijk / sqrt( sum_{i,k_spatial} (w'_ijk)^2 + eps )
        // Sum over input channels (dim 2) and spatial kernel dims (dim 3, 4). Keep C_out (dim 1) and N (dim 0).
        torch::Tensor w_demodulated_norm_sq = w_modulated.pow(2).sum({2, 3, 4}, /*keepdim=*/true); // (N, C_out, 1, 1, 1)
        torch::Tensor w_demodulated = w_modulated * torch::rsqrt(w_demodulated_norm_sq + eps_);
        // w_demodulated shape: (N, C_out, C_in, kH, kW)

        // --- 3. Perform convolution ---
        // We have N sets of weights. We need to do N convolutions or one big grouped convolution.
        // Reshape x: (N, C_in, H, W) -> (1, N*C_in, H, W)
        // Reshape w_demodulated: (N, C_out, C_in, kH, kW) -> (N*C_out, C_in, kH, kW)
        // Then use groups = N for the convolution.

        x = x.reshape({1, N * in_channels_, x.size(2), x.size(3)});
        w_demodulated = w_demodulated.reshape({N * out_channels_, in_channels_, kernel_size_[0], kernel_size_[1]});

        torch::Tensor bias_term;
        if (bias_defined_) {
            // Bias is (C_out). Need to repeat it for N*C_out.
            bias_term = bias_.repeat(N); // (N*C_out)
        }

        torch::Tensor output = torch::conv2d(x, w_demodulated, bias_term,
                                             stride_, padding_, dilation_,
                                             /*groups=*/N * groups_); // Effective groups = N * original_groups_
                                                                    // If original_groups_ was 1, then groups=N.

        // Reshape output: (1, N*C_out, H_out, W_out) -> (N, C_out, H_out, W_out)
        output = output.view({N, out_channels_, output.size(2), output.size(3)});

        return output;
    }


    void pretty_print(std::ostream& stream) const override {
        stream << "WeightDemodulatedConv2d(in_channels=" << in_channels_
               << ", out_channels=" << out_channels_
               << ", kernel_size=[" << kernel_size_[0] << "," << kernel_size_[1] << "]"
               << ", stride=[" << stride_[0] << "," << stride_[1] << "]"
               << ", padding=[" << padding_[0] << "," << padding_[1] << "]"
               << ", groups=" << groups_
               << ", bias=" << (bias_defined_ ? "true" : "false")
               << ", demod_eps=" << eps_ << ")";
    }
};


// Define the public forward method for the ModuleHolder
torch::Tensor WeightDemodulatedConv2d::forward(const torch::Tensor& x, const torch::Tensor& s_scales) {
    return impl_->forward_impl(x, s_scales);
}


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    int64_t N = 2;
    int64_t C_in = 3;
    int64_t C_out = 5;
    int64_t H_in = 8, W_in = 8;
    std::vector<long> kernel_size = {3,3};
    std::vector<long> padding = {1,1}; // To keep H,W same if stride=1

    // --- Test Case 1: Basic WeightDemodulatedConv2d (groups=1) ---
    std::cout << "--- Test Case 1: WeightDemodulatedConv2d (groups=1) ---" << std::endl;
    // For this test, groups must be 1 due to simplified logic in forward_impl
    WeightDemodulatedConv2d wd_conv1(C_in, C_out, kernel_size, 1, padding, 1, /*groups=*/1, true);
    // std::cout << wd_conv1 << std::endl;

    torch::Tensor x1 = torch::randn({N, C_in, H_in, W_in});
    // Style scales: (N, C_in). Let's make them varied.
    torch::Tensor s1 = torch::rand({N, C_in}) + 0.5; // Scales between 0.5 and 1.5
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
    std::cout << "Style scales s1 shape: " << s1.sizes() << std::endl;
    std::cout << "s1 example: " << s1[0] << std::endl;

    torch::Tensor y1 = wd_conv1->forward(x1, s1);
    std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
    // Expected output spatial size: (H_in - kH + 2*padH)/strideH + 1 = (8 - 3 + 2*1)/1 + 1 = 8
    TORCH_CHECK(y1.size(0) == N && y1.size(1) == C_out && y1.size(2) == H_in && y1.size(3) == W_in,
                "Output y1 shape mismatch!");

    // Check if output varies with style scales
    torch::Tensor s2 = torch::rand({N, C_in}) + 2.0; // Different scales
    torch::Tensor y1_alt_style = wd_conv1->forward(x1, s2);
    TORCH_CHECK(!torch::allclose(y1, y1_alt_style), "Output should change with different style scales.");
    std::cout << "Output with different style scales leads to different results: true" << std::endl;


    // --- Test Case 2: Check backward pass ---
    std::cout << "\n--- Test Case 2: Backward pass check ---" << std::endl;
    WeightDemodulatedConv2d wd_conv2(C_in, C_out, kernel_size, 1, padding, 1, 1, true);
    wd_conv2->train();

    torch::Tensor x2 = torch::randn({N, C_in, H_in, W_in}, torch::requires_grad());
    torch::Tensor s_scales2 = torch::rand({N, C_in}, torch::requires_grad()) + 0.5; // Styles also require grad

    torch::Tensor y2 = wd_conv2->forward(x2, s_scales2);
    torch::Tensor loss = y2.mean();
    loss.backward();

    bool grad_exists_x2 = x2.grad().defined() && x2.grad().abs().sum().item<double>() > 0;
    bool grad_exists_s_scales2 = s_scales2.grad().defined() && s_scales2.grad().abs().sum().item<double>() > 0;
    bool grad_exists_weight = wd_conv2->impl_->weight_.grad().defined() &&
                              wd_conv2->impl_->weight_.grad().abs().sum().item<double>() > 0;
    bool grad_exists_bias = wd_conv2->impl_->bias_.grad().defined() &&
                            wd_conv2->impl_->bias_.grad().abs().sum().item<double>() > 0;


    std::cout << "Gradient exists for x2: " << (grad_exists_x2 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for s_scales2: " << (grad_exists_s_scales2 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for conv weight: " << (grad_exists_weight ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for conv bias: " << (grad_exists_bias ? "true" : "false") << std::endl;

    TORCH_CHECK(grad_exists_x2, "No gradient for x2!");
    TORCH_CHECK(grad_exists_s_scales2, "No gradient for s_scales2!");
    TORCH_CHECK(grad_exists_weight, "No gradient for conv weight!");
    TORCH_CHECK(grad_exists_bias, "No gradient for conv bias!");


    std::cout << "\nWeightDemodulatedConv2d tests finished." << std::endl;
    std::cout << "Note: Grouped convolution (groups > 1) for weight demodulation is complex and this impl is simplified for groups=1." << std::endl;

    return 0;
}


namespace xt::norm
{
    auto WeightDemodulization::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
