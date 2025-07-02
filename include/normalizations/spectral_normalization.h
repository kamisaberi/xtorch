// #pragma once
//
// #include "common.h"
//
// namespace xt::norm
// {
//     struct SpectralNorm : xt::Module
//     {
//     public:
//         SpectralNorm(xt::Module module_to_wrap,
//                      std::string weight_name = "weight",
//                      int64_t num_power_iterations = 1,
//                      double eps = 1e-12);
//
//
//         auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//
//     private:
//         torch::nn::Module module_; // The module being wrapped (e.g., Linear, Conv2d)
//         std::string weight_name_; // Name of the weight parameter (e.g., "weight")
//         int64_t num_power_iterations_;
//         double eps_;
//
//         // 'u' vector for power iteration, registered as a buffer
//         // Its shape depends on the weight matrix. Initialized in the constructor or first forward.
//         torch::Tensor u_;
//
//         // Store the original weight parameter before spectral normalization
//         // This is not strictly needed if we recompute W_sn = W_orig / sigma on every forward pass.
//         // PyTorch's implementation often stores W_orig and computes W on the fly.
//         // For simplicity, we'll assume we operate on the module's 'weight' directly for normalization,
//         // but ideally, we'd have W_orig and W (W being W_orig / sigma).
//         // Let's try to follow PyTorch's nn.utils.spectral_norm approach:
//         // It redefines the 'weight' parameter as a computed property.
//         // We'll store the original weight as 'weight_orig' and compute 'weight' (spectrally_normalized_weight).
//     };
// }
