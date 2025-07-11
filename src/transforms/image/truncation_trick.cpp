#include "include/transforms/image/truncation_trick.h"


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