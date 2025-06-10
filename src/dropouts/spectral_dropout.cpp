#include "include/dropouts/spectral_dropout.h"





// #include <torch/torch.h>
// #include <torch/fft.h> // For FFT operations
// #include <ostream>     // For std::ostream
//
// struct SpectralDropoutImpl : torch::nn::Module {
//     double p_drop_freq_; // Probability of dropping a frequency component (after considering Hermitian symmetry)
//     double epsilon_ = 1e-7;
//
//     SpectralDropoutImpl(double p_drop_freq = 0.5) : p_drop_freq_(p_drop_freq) {
//         TORCH_CHECK(p_drop_freq_ >= 0.0 && p_drop_freq_ <= 1.0,
//                     "SpectralDropout p_drop_freq must be between 0 and 1.");
//     }
//
//     // This forward method takes a weight tensor and applies spectral dropout.
//     // It's assumed this weight_tensor is 2D for simplicity (e.g., from a Linear layer).
//     // For Conv kernels, one might apply this to each (out_channels, in_channels) slice of a 4D kernel,
//     // or to the reshaped kernel.
//     torch::Tensor forward(const torch::Tensor& weight_tensor) {
//         if (!this->is_training() || p_drop_freq_ == 0.0) {
//             return weight_tensor;
//         }
//         if (p_drop_freq_ == 1.0) {
//             // Dropping all frequencies effectively zeros out the weights (DC component might be an exception
//             // depending on exact masking, but if DC is also dropped, it's zeros).
//             // A more nuanced p_drop_freq_=1 might preserve DC or mean. For simplicity, zero out.
//             return torch::zeros_like(weight_tensor);
//         }
//
//         TORCH_CHECK(weight_tensor.dim() == 2, "SpectralDropout currently implemented for 2D weight tensors only.");
//         TORCH_CHECK(weight_tensor.is_floating_point() || weight_tensor.is_complex(), "Weight tensor must be floating point or complex for FFT.");
//
//         torch::Tensor input_weights = weight_tensor;
//         if (!input_weights.is_complex()) {
//             // FFT expects complex input, but fftn/fftn can take real and return complex.
//             // Using rfft2 for real input to complex output (half-spectrum due to Hermitian symmetry)
//         }
//
//         // 1. Compute FFT of the weights
//         // For real inputs, rfft2 gives the non-redundant half of the spectrum.
//         torch::Tensor weight_fft = torch::fft::rfft2(input_weights, /*s=*/c10::nullopt, /*dim=*/{-2, -1}, /*norm=*/"backward");
//         // norm="backward" makes ifft(fft(x)) = x. Default "forward" has 1/N scaling on ifft.
//         // "ortho" gives sqrt(1/N) on both. "backward" is common for this type of op.
//
//         // 2. Generate a dropout mask in the frequency domain
//         // The mask should be generated for the shape of weight_fft.
//         // For rfft2 output, the last dimension is (N/2 + 1).
//
//         double keep_prob = 1.0 - p_drop_freq_;
//         torch::Tensor freq_mask = torch::bernoulli(
//             torch::full_like(weight_fft, keep_prob) // Full_like works for complex by making real/imag parts same
//         ).to(weight_fft.dtype()); // Ensure mask is complex if weight_fft is complex.
//                                   // For complex, bernoulli on full_like acts on real part then casts.
//                                   // A more precise way for complex mask:
//                                   // torch::Tensor real_mask_vals = torch::bernoulli(torch::full(weight_fft.sizes(), keep_prob, weight_fft.options().dtype(input_weights.scalar_type())));
//                                   // freq_mask = torch::complex(real_mask_vals, torch::zeros_like(real_mask_vals));
//
//         // Note on Hermitian symmetry for rfft:
//         // The DC component (0,0) and Nyquist component (if present and N is even) are real.
//         // Other components have conjugates. Dropping one requires dropping its conjugate.
//         // A simple Bernoulli mask on the rfft output inherently handles this if a component
//         // and its implicit conjugate are both derived from the same Bernoulli trial.
//         // However, a more careful masking strategy would explicitly preserve/handle DC and symmetry.
//         // For this "simple" version, we apply a uniform Bernoulli mask.
//         // A common strategy is to always keep the DC component (freq_mask[0,0] = 1.0).
//         // if (weight_fft.size(0) > 0 && weight_fft.size(1) > 0) {
//         //    freq_mask.index_put_({0,0}, torch::complex(torch::tensor(1.0, weight_fft.options().dtype(input_weights.scalar_type())),
//         //                                              torch::tensor(0.0, weight_fft.options().dtype(input_weights.scalar_type())) ) );
//         // }
//
//
//         // 3. Apply the mask
//         torch::Tensor masked_weight_fft = weight_fft * freq_mask;
//
//         // 4. Compute Inverse FFT to get modified weights
//         // irfft2 expects the half-spectrum (output of rfft2) and original spatial dimensions.
//         torch::Tensor modified_weights = torch::fft::irfft2(masked_weight_fft, /*s=*/input_weights.sizes().slice(input_weights.dim()-2), /*dim=*/{-2,-1}, /*norm=*/"backward");
//
//         // Scaling: Inverted dropout scaling.
//         // The number of "effective" components in rfft2 output is roughly half the full FFT.
//         // If we drop p_drop_freq_ of these, we keep (1-p_drop_freq_).
//         // Scale by 1/keep_prob.
//         return modified_weights / (keep_prob + epsilon_);
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "SpectralDropout(p_drop_freq=" << p_drop_freq_ << ")";
//     }
// };
//
// TORCH_MODULE(SpectralDropout);
//
//
// /*
// // --- Example: How SpectralDropout might be used within a custom Linear layer ---
// // This is for context and not part of the SpectralDropoutImpl module itself.
//
// struct CustomLinearWithSpectralDropoutImpl : torch::nn::Module {
//     torch::Tensor weight_param; // The learnable weight parameter
//     torch::Tensor bias_param;
//     SpectralDropout spectral_dropout_module; // Instance of SpectralDropout
//
//     CustomLinearWithSpectralDropoutImpl(int64_t in_features, int64_t out_features, double sd_p_drop = 0.2)
//         : spectral_dropout_module(sd_p_drop) {
//         // Initialize weights (e.g., Kaiming uniform)
//         weight_param = register_parameter("weight", torch::randn({out_features, in_features}));
//         torch::nn::init::kaiming_uniform_(weight_param, std::sqrt(5)); // Example init
//
//         bias_param = register_parameter("bias", torch::randn({out_features}));
//         // Example bias init (optional)
//         if (bias_param.defined()) {
//             double bound = 1.0 / std::sqrt(in_features);
//             torch::nn::init::uniform_(bias_param, -bound, bound);
//         }
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         torch::Tensor current_weight = weight_param;
//
//         if (this->is_training()) {
//             // Apply Spectral Dropout to a copy of the weights during training
//             current_weight = spectral_dropout_module(weight_param);
//         }
//         // During evaluation, current_weight remains the original 'this->weight_param'.
//
//         return torch::nn::functional::linear(input, current_weight, bias_param);
//     }
// };
// TORCH_MODULE(CustomLinearWithSpectralDropout);
//
//
// #include <iostream>
// #include <iomanip> // For std::fixed, std::setprecision
//
// void run_spectral_dropout_example() {
//     torch::manual_seed(0);
//     std::cout << std::fixed << std::setprecision(4);
//
//     // Test SpectralDropout module directly on a weight tensor
//     SpectralDropout sd_op(0.5); // Drop 50% of frequency components
//     sd_op->train(); // Set to training mode
//
//     torch::Tensor W = torch::randn({4, 6}); // A 2D weight matrix
//     std::cout << "Original W (sum): " << W.sum().item<float>() << std::endl;
//     // std::cout << "Original W:\n" << W << std::endl;
//
//     torch::Tensor W_spectral_dropped = sd_op(W);
//     std::cout << "W after Spectral Dropout (train, sum): " << W_spectral_dropped.sum().item<float>() << std::endl;
//     // std::cout << "W_spectral_dropped:\n" << W_spectral_dropped << std::endl;
//     // The sum will change due to dropped frequencies and scaling.
//     // The structure of W will also change.
//
//     sd_op->eval(); // Set to evaluation mode
//     torch::Tensor W_eval = sd_op(W);
//     std::cout << "W after Spectral Dropout (eval, sum): " << W_eval.sum().item<float>() << std::endl;
//     TORCH_CHECK(torch::allclose(W, W_eval), "SpectralDropout eval output should be original weight if p_drop=0 or in eval.");
//     // Wait, in eval mode, SpectralDropout should return the original weights if implemented as pass-through.
//     // My current implementation has it as pass-through. If it were to apply expected value of mask, it would differ.
//     // The common use of SpectralDropout is to modify weights only during training.
//
//     // Using the custom layer that incorporates SpectralDropout
//     std::cout << "\n--- CustomLinearWithSpectralDropout Test ---" << std::endl;
//     CustomLinearWithSpectralDropout custom_layer(6, 4, 0.3); // in=6, out=4, p_drop_freq=0.3
//     torch::Tensor layer_input = torch::randn({2, 6}); // Batch=2, InFeatures=6
//
//     custom_layer->train();
//     torch::Tensor layer_output_train = custom_layer(layer_input);
//     std::cout << "CustomLayer output (train) sum: " << layer_output_train.sum().item<float>() << std::endl;
//     // (Internally, the weights used for linear operation were spectrally dropped)
//
//     custom_layer->eval();
//     torch::Tensor layer_output_eval = custom_layer(layer_input);
//     std::cout << "CustomLayer output (eval) sum: " << layer_output_eval.sum().item<float>() << std::endl;
//     // (Internally, the original weights were used)
//
//     // Compare train and eval outputs for the custom layer. They should differ.
//     torch::Tensor original_weights_output = torch::nn::functional::linear(layer_input, custom_layer->weight_param, custom_layer->bias_param);
//     TORCH_CHECK(torch::allclose(original_weights_output, layer_output_eval), "Custom layer eval mode mismatch.");
//     // Check if training output is different (it should be, due to stochastic spectral dropout)
//     // This is a weak check, as they could be coincidentally close for one run.
//     TORCH_CHECK(!torch::allclose(layer_output_train, layer_output_eval, 1e-3, 1e-3), "Custom layer train output should differ from eval.");
//
//
//     // Test with p_drop_freq = 0.0 (no spectral dropout)
//     SpectralDropout sd_op_no_drop(0.0);
//     sd_op_no_drop->train();
//     torch::Tensor W_no_spectral_drop = sd_op_no_drop(W);
//     std::cout << "\nW after Spectral Dropout (p_drop_freq=0.0, train, sum): " << W_no_spectral_drop.sum().item<float>() << std::endl;
//     TORCH_CHECK(torch::allclose(W, W_no_spectral_drop), "SpectralDropout with p_drop_freq=0.0 should be identity.");
// }
//
// // int main() {
// //    run_spectral_dropout_example();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    torch::Tensor spectral_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto SpectralDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::spectral_dropout(torch::zeros(10));
    }
}
