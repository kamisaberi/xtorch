#include "include/normalizations/attentive_normalization.h"



// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct AttentiveNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct AttentiveNormalization : torch::nn::ModuleHolder<AttentiveNormalizationImpl> {
//     using torch::nn::ModuleHolder<AttentiveNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct
// struct AttentiveNormalizationImpl : torch::nn::Module {
//     int64_t num_features_;
//     double eps_;
//     int64_t num_norm_candidates_ = 3; // IN, LN, BN
//     int64_t attention_reduction_ratio_;
//
//     // Normalization layers
//     torch::nn::InstanceNorm2d instance_norm_{nullptr};
//     torch::nn::LayerNorm layer_norm_{nullptr}; // LayerNorm needs normalized_shape
//     torch::nn::BatchNorm2d batch_norm_{nullptr};
//
//     // Attention Gate layers
//     torch::nn::AdaptiveAvgPool2d avg_pool_{nullptr};
//     torch::nn::Conv2d fc1_{nullptr};
//     torch::nn::ReLU relu_{nullptr};
//     torch::nn::Conv2d fc2_{nullptr};
//     torch::nn::Softmax softmax_{nullptr}; // Softmax along the candidates dimension
//
//     AttentiveNormalizationImpl(int64_t num_features, double eps = 1e-5, int64_t attention_reduction_ratio = 8)
//         : num_features_(num_features), eps_(eps), attention_reduction_ratio_(attention_reduction_ratio) {
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//         TORCH_CHECK(attention_reduction_ratio_ > 0, "attention_reduction_ratio must be positive.");
//
//         // 1. Initialize Normalization Layers
//         // InstanceNorm2d: affine=true by default is good, as AN combines already-affined outputs.
//         instance_norm_ = torch::nn::InstanceNorm2d(
//             torch::nn::InstanceNorm2dOptions(num_features_).eps(eps_).affine(true).track_running_stats(false)
//         );
//         register_module("instance_norm", instance_norm_);
//
//         // LayerNorm: normalized_shape should cover C, H, W for input (N, C, H, W)
//         // This means it normalizes over all feature dimensions per batch instance.
//         // If H and W are not fixed, this is problematic. Let's assume they are somewhat fixed for the LN setup
//         // or LayerNorm will normalize over C and whatever H,W the input has.
//         // The paper applies LN to the entire feature map (C,H,W) for each batch element.
//         // So, normalized_shape should be {C, H, W}. But H,W can vary.
//         // A common way for images is to normalize over channels for each spatial location (like GroupNorm with groups=1).
//         // However, the original AN paper implies normalizing (C,H,W).
//         // If we want LN to normalize over C, H, W, we need to know H, W at construction or make LN dynamic.
//         // PyTorch's LayerNorm expects normalized_shape from the *last* dimensions.
//         // For an input (N,C,H,W), if we want to normalize over C,H,W, we'd need to permute to (N,H,W,C)
//         // and specify normalized_shape={C}. Then permute back. This is common.
//         // OR: use a lambda module for LN that flattens C,H,W, applies LN on the flattened dim, then unflattens.
//         // Simpler for now: LayerNorm over the channel dimension (like GroupNorm(1, C)).
//         // However, the paper's LN normalizes over C,H,W.
//         // Let's create LN assuming the input H,W will be passed to forward and build normalized_shape there.
//         // For LibTorch, LayerNormOptions needs a std::vector<int64_t> for normalized_shape.
//         // The most straightforward LN for (N,C,H,W) that acts like the paper's LN for each N is to
//         // normalize over the last 3 dimensions if the input is always (N,C,H,W).
//         // This is tricky if H,W are not fixed at construction.
//         // For simplicity, we'll make a LayerNorm that normalizes over C, assuming H and W are spatial.
//         // To truly match paper, we'd need to know H and W for layer_norm_options
//         // A common LibTorch pattern for images with LayerNorm is to normalize (N, C, H*W) then reshape,
//         // or apply it channel-wise (like GroupNorm with num_groups=1).
//         // The paper seems to apply LN across (C,H,W) for each sample in the batch.
//         // We'll use LayerNorm with elementwise_affine=true. normalized_shape will be set dynamically in forward.
//         // This means layer_norm_ itself cannot be fully constructed here without H,W.
//         // Alternative: The paper's figure suggests LN(x_n) where x_n is the full (C,H,W) feature map for sample n.
//         // This can be done by reshaping (N, C, H, W) -> (N, C*H*W), apply LN to last dim, then reshape back.
//
//         // BatchNorm2d
//         batch_norm_ = torch::nn::BatchNorm2d(
//             torch::nn::BatchNorm2dOptions(num_features_).eps(eps_).affine(true).track_running_stats(true)
//         );
//         register_module("batch_norm", batch_norm_);
//
//         // 2. Initialize Attention Gate
//         avg_pool_ = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
//         register_module("avg_pool", avg_pool_);
//
//         int64_t hidden_features = std::max(static_cast<int64_t>(1), num_features_ / attention_reduction_ratio_);
//
//         fc1_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(num_features_, hidden_features, 1).bias(false));
//         register_module("fc1", fc1_);
//
//         relu_ = torch::nn::ReLU();
//         register_module("relu", relu_);
//
//         // Output K attention scores PER CHANNEL. So, output channels = K * num_features_
//         fc2_ = torch::nn::Conv2d(torch::nn::Conv2dOptions(hidden_features, num_norm_candidates_ * num_features_, 1).bias(false));
//         register_module("fc2", fc2_);
//
//         // Softmax over the 'candidates' dimension.
//         // After fc2, shape is (N, K*C, 1, 1). We reshape to (N, K, C, 1, 1) then softmax over dim 1.
//         softmax_ = torch::nn::Softmax(1); // To be applied on reshaped tensor
//         // No need to register_module for functional softmax, but nn::Softmax is a module
//         register_module("softmax", softmax_);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x: (N, C, H, W)
//         TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got ", x.dim());
//         TORCH_CHECK(x.size(1) == num_features_, "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
//
//         const int64_t N = x.size(0);
//         const int64_t C = x.size(1); // num_features_
//         const int64_t H = x.size(2);
//         const int64_t W = x.size(3);
//
//         // --- 1. Compute Normalization Candidates ---
//         torch::Tensor in_out = instance_norm_->forward(x);
//         torch::Tensor bn_out = batch_norm_->forward(x);
//
//         // LayerNorm: Normalize over C, H, W for each N.
//         // Reshape to (N, C*H*W), apply LN over the last dimension, then reshape back.
//         // This is how PyTorch's nn.LayerNorm([C,H,W]) would behave if C,H,W were fixed.
//         auto ln_input_shape = x.sizes();
//         torch::Tensor ln_flat = x.reshape({N, -1}); // (N, C*H*W)
//         // Dynamically create LayerNorm options if H,W change or use one that normalizes the whole flattened vector
//         auto layer_norm_options = torch::nn::LayerNormOptions({C * H * W}).eps(eps_).elementwise_affine(true);
//         torch::nn::LayerNorm dynamic_ln(layer_norm_options);
//         // Move LN to the same device as input and copy parameters if it had them (it has if affine=true)
//         // For simplicity, re-initialize. If this LN had its own state, it would need careful handling.
//         // Or, we can initialize layer_norm_ in constructor with a very large dummy shape and hope it works, or adapt.
//         // Best for true LayerNorm(shape) behavior is to instantiate here or make it a functional call.
//         // Let's use a temporary LayerNorm module.
//         // Note: if LN needs to be learnable and consistent across calls, this dynamic instantiation is not ideal.
//         //       One would need to register its parameters.
//         //       For AN, often the sub-normalizers ARE learnable.
//         //       A workaround for LibTorch: if H,W are truly dynamic, one might define LN to act on (N, C, H*W)
//         //       and normalize the last dim, then reshape.
//         if (!layer_norm_) { // Initialize on first forward if H, W needed and not fixed
//              layer_norm_ = torch::nn::LayerNorm(torch::nn::LayerNormOptions({C,H,W}).eps(eps_).elementwise_affine(true));
//              layer_norm_->to(x.device()); // Move to device
//              register_module("layer_norm", layer_norm_); // Register if created here
//         }
//         // If LayerNorm was defined to take {C,H,W} and these are variable, the above init in constructor is an issue.
//         // For this example, let's assume `layer_norm_` can handle arbitrary H, W if its `normalized_shape` only refers to `C`.
//         // To normalize over (C,H,W) as the paper implies:
//         torch::Tensor ln_out_flat = torch::layer_norm(x.reshape({N, C*H*W}), {C*H*W},
//                                                       instance_norm_->weight.defined() ? torch::ones({C*H*W}, x.options()) : torch::Tensor(), // dummy gamma if IN has none
//                                                       instance_norm_->bias.defined() ? torch::zeros({C*H*W}, x.options()) : torch::Tensor(), // dummy beta
//                                                       eps_); // Using IN's affine params as placeholders for LN's if LN needs them.
//                                                       // This is not quite right if LN has its own independent affine params.
//                                                       // A proper LN would need its own registered affine params.
//         // Let's redefine LN to have its own parameters.
//         // We'll use the functional `torch::layer_norm`. We need to provide weight and bias.
//         // If layer_norm_ was properly constructed for (C,H,W) and H,W are fixed:
//         // torch::Tensor ln_out = layer_norm_->forward(x);
//         // Using a simpler LN for now: normalize over C channels only, for each spatial location (H,W)
//         // This is equivalent to GroupNorm(1, C).
//         // For (N,C,H,W) -> LayerNorm({C}, elementwise_affine=true) -> (N,C,H,W)
//         // This is not what the original AN paper's LN does. It normalizes the whole (C,H,W) feature map.
//         // To achieve original AN's LN:
//         // 1. Store LN affine parameters as `ln_gamma` (shape C*H*W) and `ln_beta` (shape C*H*W).
//         //    This is problematic if H,W vary.
//         // 2. A more flexible LN that is common is to permute x to (N, H, W, C) and apply LN over last dim C.
//         // For simplicity and to avoid H,W dependency in LN parameters:
//         // Use a version of LayerNorm that reshapes, normalizes, and reshapes back.
//         // This specific LN implementation is a bit of a simplification due to LibTorch LayerNorm requiring fixed shape.
//         // Let's assume a simplified LayerNorm that normalizes over the C dimension for each H,W independently:
//         // This is essentially GroupNorm with num_groups = C. Or LayerNorm after permuting (N,H,W,C) applied to C.
//
//         // Re-evaluating LN approach for (N,C,H,W) to match paper's intent (normalize full C,H,W features per sample):
//         // We need parameters for LN if affine. If H,W vary, the parameter size varies. This is the main issue.
//         // If we assume H,W are fixed for the model, then LN can be:
//         if (!layer_norm_) { // Initialize LayerNorm with fixed C,H,W shape
//             layer_norm_ = torch::nn::LayerNorm(torch::nn::LayerNormOptions({C, H, W}).eps(eps_).elementwise_affine(true));
//             this->register_module("layer_norm_dynamic", layer_norm_); // register if created here
//             layer_norm_->to(x.device());
//         }
//         torch::Tensor ln_out;
//         if (layer_norm_->options.normalized_shape() == std::vector<int64_t>{C,H,W}) {
//              ln_out = layer_norm_->forward(x);
//         } else {
//             // Fallback or error if H,W mismatch pre-configured LN
//             // For this example, let's make a strong assumption that H,W are fixed for LN.
//             // If they are not, one must use a different LN strategy.
//              TORCH_WARN("LayerNorm configured with different H,W than input. Re-normalizing C dimension only as fallback.");
//              // Fallback: LayerNorm over channels for each spatial point (N, H, W, C) -> normalize C
//              auto x_permuted = x.permute({0, 2, 3, 1}).contiguous(); // (N, H, W, C)
//              auto temp_ln_options = torch::nn::LayerNormOptions({C}).eps(eps_).elementwise_affine(true);
//              // Need to manage parameters for this LN if affine is true.
//              // For now, let's use functional with no affine for this fallback.
//              ln_out = torch::layer_norm(x_permuted, {C}, {}, {}, eps_).permute({0, 3, 1, 2}); // (N,C,H,W)
//         }
//
//
//         // --- 2. Compute Attention Weights ---
//         torch::Tensor att_scores = avg_pool_->forward(x); // (N, C, 1, 1)
//         att_scores = fc1_->forward(att_scores);       // (N, hidden_C, 1, 1)
//         att_scores = relu_->forward(att_scores);      // (N, hidden_C, 1, 1)
//         att_scores = fc2_->forward(att_scores);       // (N, K*C, 1, 1)
//
//         // Reshape for softmax: (N, K*C, 1, 1) -> (N, K, C, 1, 1)
//         // K = num_norm_candidates_
//         att_scores = att_scores.view({N, num_norm_candidates_, num_features_, 1, 1});
//         torch::Tensor attention_weights = softmax_->forward(att_scores); // Softmax along dim 1 (K)
//                                                                     // Shape: (N, K, C, 1, 1)
//
//         // --- 3. Combine Candidates ---
//         // Stack candidates: (N, C, H, W) for each -> (K, N, C, H, W) then permute to (N, K, C, H, W)
//         // Or stack directly into desired shape if possible.
//         torch::Tensor candidates = torch::stack({in_out, ln_out, bn_out}, /*dim=*/1); // (N, K, C, H, W)
//
//         // Element-wise multiply attention_weights with candidates.
//         // attention_weights: (N, K, C, 1, 1)
//         // candidates:        (N, K, C, H, W)
//         // Broadcasting will make weights apply to H, W dimensions.
//         torch::Tensor weighted_candidates = candidates * attention_weights;
//
//         // Sum along the candidates dimension (dim 1)
//         torch::Tensor output = weighted_candidates.sum(/*dim=*/1); // (N, C, H, W)
//
//         return output;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "AttentiveNormalization(num_features=" << num_features_
//                << ", eps=" << eps_
//                << ", num_candidates=" << num_norm_candidates_
//                << ", attention_reduction=" << attention_reduction_ratio_ << ")";
//         if (layer_norm_) {
//             stream << "\n  LayerNorm options: normalized_shape=[";
//             for(size_t i=0; i<layer_norm_->options.normalized_shape().size(); ++i) {
//                 stream << layer_norm_->options.normalized_shape()[i] << (i == layer_norm_->options.normalized_shape().size()-1 ? "" : ", ");
//             }
//             stream << "], eps=" << layer_norm_->options.eps() << ", affine=" << layer_norm_->options.elementwise_affine();
//         } else {
//             stream << "\n  LayerNorm not fully initialized (depends on first input's H,W)";
//         }
//     }
// };
// TORCH_MODULE(AttentiveNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 64;
//     int64_t N = 4, H = 16, W = 16; // Fixed H, W for LayerNorm proper initialization
//
//     // Important: LayerNorm in this setup is initialized assuming fixed H,W from the first forward pass,
//     // or if its options are set to only normalize over C.
//     // The current AttentiveNormalizationImpl constructor initializes LN requiring fixed H,W for its params.
//     // So, we construct it with num_features, and then on first forward, LN will be fully built.
//     AttentiveNormalization an_module(num_features);
//     // std::cout << an_module << std::endl; // Parameter print
//
//     // Move module to device if using CUDA
//     // torch::Device device(torch::kCUDA_IF_AVAILABLE);
//     // an_module->to(device);
//
//     // Create a dummy input tensor
//     torch::Tensor input = torch::randn({N, num_features, H, W}); // .to(device);
//     std::cout << "Input shape: " << input.sizes() << std::endl;
//
//     // --- Training mode (for BatchNorm mainly) ---
//     an_module->train();
//     std::cout << "\n--- Training mode forward pass ---" << std::endl;
//     torch::Tensor output_train = an_module->forward(input);
//     std::cout << "Output_train shape: " << output_train.sizes() << std::endl;
//     std::cout << "Output_train mean (all): " << output_train.mean().item<double>() << std::endl;
//     std::cout << "Output_train std (all): " << output_train.std().item<double>() << std::endl;
//
//     // Access attention weights (requires modifying the forward method or using hooks)
//     // For example, if we wanted to see the attention weights for the first norm type (IN) for the first sample, first channel:
//     // This is a bit tricky without direct access.
//     // We can infer by checking the outputs of submodules if needed for debugging.
//     // E.g. an_module->instance_norm->forward(input)
//
//     // --- Evaluation mode (for BatchNorm mainly) ---
//     an_module->eval();
//     std::cout << "\n--- Evaluation mode forward pass ---" << std::endl;
//     torch::Tensor output_eval = an_module->forward(input); // BN will use running stats
//     std::cout << "Output_eval shape: " << output_eval.sizes() << std::endl;
//     std::cout << "Output_eval mean (all): " << output_eval.mean().item<double>() << std::endl;
//     std::cout << "Output_eval std (all): " << output_eval.std().item<double>() << std::endl;
//
//     // Compare train and eval outputs (should differ due to BatchNorm)
//     bool different = !torch::allclose(output_train, output_eval);
//     std::cout << "Output_train and Output_eval are different: " << (different ? "true" : "false") << std::endl;
//     TORCH_CHECK(different, "Train and Eval outputs should differ due to BatchNorm state changes.");
//
//     // Print module structure including LayerNorm details after first forward
//     std::cout << "\nModule structure after initialization:" << std::endl;
//     std::cout << *an_module << std::endl;
//
//
//     // Test with different H, W - this will trigger the LayerNorm fallback warning
//     // if the LN was configured for specific H,W and those H,W change.
//     std::cout << "\n--- Test with different H, W ---" << std::endl;
//     torch::Tensor input_diff_hw = torch::randn({N, num_features, H/2, W/2});
//     try {
//         an_module->eval(); // Keep in eval mode
//         torch::Tensor output_diff_hw = an_module->forward(input_diff_hw);
//         std::cout << "Output_diff_hw shape: " << output_diff_hw.sizes() << std::endl;
//     } catch (const c10::Error& e) {
//         std::cerr << "Error with different H,W: " << e.what() << std::endl;
//         std::cerr << "This might be expected if LayerNorm was strictly configured for specific H,W "
//                   << "and the fallback is not perfectly robust or not desired." << std::endl;
//     }
//     std::cout << "Attentive Normalization example finished." << std::endl;
//
//     return 0;
// }




namespace xt::norm
{
    auto AttentiveNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
