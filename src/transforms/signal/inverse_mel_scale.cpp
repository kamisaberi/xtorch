#include <transforms/signal/inverse_mel_scale.h>


/*
// The example usage from before remains valid.
// This improved implementation is a drop-in replacement.
*/

namespace xt::transforms::signal {

    // Helper functions for Mel scale conversion (no changes needed here)
    inline double hz_to_mel(double hz) {
        return 2595.0 * std::log10(1.0 + hz / 700.0);
    }

    inline double mel_to_hz(double mel) {
        return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
    }

    // --- FULLY REVISED AND VECTORIZED FUNCTION ---
    torch::Tensor InverseMelScale::create_mel_filter_banks(
        int n_stft, int n_mels, int sample_rate, double f_min, double f_max) {

        // --- Fix 1: Removed c10::fmap and data_ptr, using direct tensor math ---

        // 1. Define frequency range in Mel scale and create evenly spaced points
        double min_mel = hz_to_mel(f_min);
        double max_mel = hz_to_mel(f_max);
        auto mel_points = torch::linspace(min_mel, max_mel, n_mels + 2);

        // 2. Convert Mel points back to Hz using element-wise tensor operations
        // This is the direct fix for the original bug. It's safe and clear.
        auto hz_points = 700.0 * (torch::pow(10.0, mel_points / 2595.0) - 1.0);

        // 3. Convert Hz points to FFT bin indices
        int n_fft = (n_stft - 1) * 2;
        auto fft_bins = torch::floor((n_fft + 1) * hz_points / sample_rate);

        // --- Improvement: Vectorized filter bank creation (replaces C++ loop) ---

        // 4. Create tensors for the start, center, and end of each triangular filter
        auto f_start = fft_bins.slice(0, 0, n_mels);
        auto f_center = fft_bins.slice(0, 1, n_mels + 1);
        auto f_end = fft_bins.slice(0, 2, n_mels + 2);

        // 5. Calculate the left and right slopes of the filters using broadcasting
        // Reshape bin tensors to (n_mels, 1) and freq axis to (1, n_stft)
        auto all_freqs = torch::arange(0, n_stft).unsqueeze(0);
        f_start = f_start.unsqueeze(1);
        f_center = f_center.unsqueeze(1);
        f_end = f_end.unsqueeze(1);

        auto left_ramp = (all_freqs - f_start) / (f_center - f_start);
        auto right_ramp = (f_end - all_freqs) / (f_end - f_center);

        // 6. Combine ramps to form triangles and clamp to zero. This is a common
        // and efficient pattern for creating triangular filter banks.
        auto mel_basis = torch::max(
            torch::zeros({1}),
            torch::min(left_ramp, right_ramp)
        );

        return mel_basis;
    }

    InverseMelScale::InverseMelScale(
        int n_stft,
        int n_mels,
        int sample_rate,
        double f_min,
        c10::optional<double> f_max_opt) {

        double f_max = f_max_opt.value_or(static_cast<double>(sample_rate) / 2.0);

        // Create the Mel filter banks using the robust, vectorized function
        torch::Tensor mel_basis = create_mel_filter_banks(n_stft, n_mels, sample_rate, f_min, f_max);

        // --- Improvement: Use proper pseudo-inverse for better accuracy ---
        // Instead of a simple normalized transpose, torch.linalg.pinverse gives a
        // more numerically stable result for inverting the Mel basis.
        inverse_mel_basis_ = xt::linalg::pinverse(mel_basis).clone();
    }

    auto InverseMelScale::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- No changes needed here, the logic was already correct ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("InverseMelScale::forward received an empty list.");
        }
        torch::Tensor mel_spec = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!mel_spec.defined()) {
            throw std::invalid_argument("Input tensor passed to InverseMelScale is not defined.");
        }

        // The core operation is a matrix multiplication.
        // Shapes: (n_stft, n_mels) @ (..., n_mels, time) -> (..., n_stft, time)
        torch::Tensor linear_spec = torch::matmul(
            inverse_mel_basis_.to(mel_spec.options()), // Use .options() for safety
            mel_spec
        );

        // Ensure the result is non-negative, as it represents power.
        return torch::clamp_min(linear_spec, 0.0);
    }

} // namespace xt::transforms::signal