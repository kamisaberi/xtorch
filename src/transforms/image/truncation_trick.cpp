#include <transforms/image/truncation_trick.h>

//
// #include "transforms/general/truncation_trick.h"
// #include <iostream>
//
// // --- Dummy Mapping Network for Demonstration ---
// struct MappingNetwork : torch::nn::Module {
//     torch::nn::Linear layer;
//     MappingNetwork(int z_dim, int w_dim) : layer(z_dim, w_dim) {
//         register_module("layer", layer);
//     }
//     torch::Tensor forward(torch::Tensor z) {
//         return layer->forward(z);
//     }
// };
//
// int main() {
//     // --- 1. Setup ---
//     int batch_size = 4;
//     int z_dim = 512;
//     int w_dim = 512;
//
//     // --- 2. Pre-compute or load the average latent vector (w_avg) ---
//     // In a real scenario, you would calculate this by running thousands of
//     // random z vectors through your mapping network and averaging the results.
//     // For this demo, we'll just create a dummy one.
//     torch::Tensor w_avg = torch::randn({1, w_dim}) * 0.1; // Typically close to zero
//
//     // --- 3. Instantiate the transform with the desired truncation level ---
//     double truncation_psi = 0.7;
//     xt::transforms::general::TruncationTrick truncator(w_avg, truncation_psi);
//
//     // --- 4. Simulate a generation step ---
//     MappingNetwork mapping_network(z_dim, w_dim);
//     torch::Tensor z = torch::randn({batch_size, z_dim}); // New random codes
//     torch::Tensor w = mapping_network.forward(z);        // Map to style space
//
//     std::cout << "Original distance from avg (L2 norm): "
//               << torch::linalg::norm(w - w_avg, c10::nullopt, c10::IntArrayRef({1})).mean().item<float>()
//               << std::endl;
//
//     // --- 5. Apply the TruncationTrick ---
//     std::any result_any = truncator.forward({w});
//     torch::Tensor w_truncated = std::any_cast<torch::Tensor>(result_any);
//
//     // --- 6. Check the output ---
//     std::cout << "Truncated distance from avg (L2 norm): "
//               << torch::linalg::norm(w_truncated - w_avg, c10::nullopt, c10::IntArrayRef({1})).mean().item<float>()
//               << std::endl;
//     // The truncated distance should be smaller than the original distance by a factor of ~psi.
//
//     std::cout << "\nShape of original latents:    " << w.sizes() << std::endl;
//     std::cout << "Shape of truncated latents: " << w_truncated.sizes() << std::endl;
//
//     // Now you would feed `w_truncated` into the GAN's synthesis network to generate high-quality images.
//
//     return 0;
// }

namespace xt::transforms::general {

    TruncationTrick::TruncationTrick() : truncation_psi_(0.7) {}

    TruncationTrick::TruncationTrick(torch::Tensor w_avg, double truncation_psi)
        : w_avg_(w_avg), truncation_psi_(truncation_psi) {

        if (!w_avg_.defined()) {
            throw std::invalid_argument("Average latent vector (w_avg) must be a defined tensor.");
        }
        if (truncation_psi_ < 0.0) {
            // A value > 1.0 is allowed; it extrapolates instead of interpolates.
            // But we can warn the user.
        }

        // Ensure w_avg is 2D [1, LatentDim] for easy broadcasting
        if (w_avg_.dim() == 1) {
            w_avg_ = w_avg_.unsqueeze(0);
        }
    }

    auto TruncationTrick::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("TruncationTrick::forward received an empty list of tensors.");
        }
        torch::Tensor w = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!w.defined()) {
            throw std::invalid_argument("Input latent vector (w) is not defined.");
        }
        if (w.dim() != 2) {
            throw std::invalid_argument("TruncationTrick expects a 2D batch of latent vectors [B, LatentDim].");
        }
        if (w.size(1) != w_avg_.size(1)) {
            throw std::invalid_argument("Dimension of input latent vector does not match the average latent vector.");
        }

        // If psi is 1.0, there is no truncation.
        if (truncation_psi_ == 1.0) {
            return w;
        }

        // Move w_avg to the same device as the input w
        w_avg_ = w_avg_.to(w.device());

        // 2. --- Apply the Truncation Trick ---
        // Formula: w_truncated = w_avg + psi * (w - w_avg)
        // This is a linear interpolation (lerp).
        torch::Tensor w_truncated = w_avg_ + truncation_psi_ * (w - w_avg_);

        return w_truncated;
    }

} // namespace xt::transforms::general