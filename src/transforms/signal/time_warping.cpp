#include "include/transforms/signal/time_warping.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load an audio file.
    auto [waveform, sample_rate] = xt::utils::audio::load("some_audio_file.wav");
    std::cout << "Original waveform length: " << waveform.size(0) << std::endl;

    // 2. Create the TimeWarping transform.
    // Allow the center point to be shifted by up to 8% of the total length.
    xt::transforms::signal::TimeWarping time_warper(0.08);

    // 3. Apply the transform.
    torch::Tensor warped_waveform = std::any_cast<torch::Tensor>(
        time_warper.forward({waveform})
    );

    // 4. Check the new length (should be the same) and save the result.
    std::cout << "Warped waveform length: " << warped_waveform.size(0) << std::endl;
    xt::utils::audio::save("time_warped.wav", warped_waveform, sample_rate);

    return 0;
}
*/

namespace { // Anonymous namespace for local helper functions

// Helper for 1D linear interpolation on a 1D tensor.
    torch::Tensor linear_interp1d(const torch::Tensor& data, const torch::Tensor& indices) {
        long length = data.size(0);
        auto indices_floor = torch::floor(indices).to(torch::kLong);
        auto indices_ceil = torch::ceil(indices).to(torch::kLong);
        auto alpha = (indices - indices_floor);

        // Clamp indices to be within bounds
        indices_floor = torch::clamp(indices_floor, 0, length - 1);
        indices_ceil = torch::clamp(indices_ceil, 0, length - 1);

        auto data_floor = data.index({indices_floor});
        auto data_ceil = data.index({indices_ceil});

        return (1.0 - alpha) * data_floor + alpha * data_ceil;
    }

} // namespace


namespace xt::transforms::signal {

    TimeWarping::TimeWarping(double max_time_warp_percent, double p)
            : max_time_warp_percent_(max_time_warp_percent), p_(p) {

        if (max_time_warp_percent_ < 0.0 || max_time_warp_percent_ >= 0.5) {
            throw std::invalid_argument("max_time_warp_percent must be in [0, 0.5).");
        }

        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto TimeWarping::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("TimeWarping::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_) {
            return waveform;
        }

        long L = waveform.size(-1);
        if (L < 2) {
            return waveform; // Cannot warp a signal that is too short
        }

        // --- 2. Define Anchor Points and Warp ---
        // The anchor point is the middle of the signal.
        long t_anchor = L / 2;

        // Determine how far to shift the anchor point.
        long max_warp = static_cast<long>(L * max_time_warp_percent_);
        std::uniform_int_distribution<long> warp_dist(-max_warp, max_warp);
        long warp_amount = warp_dist(random_engine_);

        // The new position of the anchor point.
        long t_warped_anchor = t_anchor + warp_amount;

        // Handle edge cases to prevent division by zero.
        if (t_warped_anchor <= 0 || t_warped_anchor >= L) {
            return waveform;
        }

        // --- 3. Create the Non-linear Time Mapping ---
        // We generate a list of source indices to sample from the original waveform.
        auto new_time_axis = torch::arange(0, L, waveform.options());

        // Calculate the two different scaling factors.
        double rate1 = static_cast<double>(t_anchor) / t_warped_anchor;
        double rate2 = static_cast<double>(L - t_anchor) / (L - t_warped_anchor);

        // Use `torch::where` for an efficient, vectorized implementation.
        // For t < t_warped_anchor: source_idx = t * rate1
        // For t >= t_warped_anchor: source_idx = t_anchor + (t - t_warped_anchor) * rate2
        torch::Tensor source_indices = torch::where(
                new_time_axis < t_warped_anchor,
                new_time_axis * rate1,
                t_anchor + (new_time_axis - t_warped_anchor) * rate2
        );

        // --- 4. Resample the waveform using the new time map ---
        torch::Tensor warped_waveform = linear_interp1d(waveform, source_indices);

        return warped_waveform;
    }

} // namespace xt::transforms::signal