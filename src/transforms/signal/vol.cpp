#include <transforms/signal/vol.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load an audio file.
    auto [waveform, sample_rate] = xt::utils::audio::load("some_audio_file.wav");
    std::cout << "Max amplitude before: " << torch::max(torch::abs(waveform)).item<float>() << std::endl;

    // 2. Create the Vol transform.
    // This will randomly apply a gain between -10dB (quieter) and +5dB (louder).
    xt::transforms::signal::Vol vol_transform(-10.0, 5.0);

    // 3. Apply the transform.
    torch::Tensor adjusted_waveform = std::any_cast<torch::Tensor>(
        vol_transform.forward({waveform})
    );

    // 4. Check the new amplitude and save the result.
    std::cout << "Max amplitude after: " << torch::max(torch::abs(adjusted_waveform)).item<float>() << std::endl;
    xt::utils::audio::save("volume_adjusted.wav", adjusted_waveform, sample_rate);

    return 0;
}
*/

namespace xt::transforms::signal {

    Vol::Vol(double min_db, double max_db, double p)
            : min_db_(min_db), max_db_(max_db), p_(p) {

        if (min_db_ > max_db_) {
            throw std::invalid_argument("min_db cannot be greater than max_db.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto Vol::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Vol::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_) {
            return waveform;
        }

        // --- 2. Choose Random dB Level and Convert to Linear Factor ---
        std::uniform_real_distribution<double> db_dist(min_db_, max_db_);
        double db = db_dist(random_engine_);

        // Convert decibels to a linear amplitude scaling factor.
        // Formula: factor = 10^(dB / 20)
        double scaling_factor = std::pow(10.0, db / 20.0);

        // --- 3. Apply Scaling and Clamp to Prevent Clipping ---
        // Multiply the waveform by the scaling factor.
        torch::Tensor scaled_waveform = waveform * scaling_factor;

        // Clamp the output to the valid [-1.0, 1.0] range to prevent distortion
        // when saving to integer-based audio formats (like 16-bit WAV).
        torch::Tensor clamped_waveform = torch::clamp(scaled_waveform, -1.0, 1.0);

        return clamped_waveform;
    }

} // namespace xt::transforms::signal