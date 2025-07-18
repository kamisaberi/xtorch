#include <transforms/signal/speed_perturbation.h>



/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load an audio file.
    auto [waveform, sample_rate] = xt::utils::audio::load("some_audio_file.wav");
    std::cout << "Original waveform length: " << waveform.size(0) << std::endl;

    // 2. Create the SpeedPerturbation transform.
    // It will randomly choose a speed between 0.9 (10% slower) and 1.1 (10% faster).
    xt::transforms::signal::SpeedPerturbation speed_perturber(sample_rate, 0.9, 1.1);

    // 3. Apply the transform.
    torch::Tensor perturbed_waveform = std::any_cast<torch::Tensor>(
        speed_perturber.forward({waveform})
    );

    // 4. Check the new length and save the result.
    std::cout << "Perturbed waveform length: " << perturbed_waveform.size(0) << std::endl;
    xt::utils::audio::save("speed_perturbed.wav", perturbed_waveform, sample_rate);

    return 0;
}
*/

namespace { // Anonymous namespace for local helper functions

// Helper to wrap phase values to the [-PI, PI] range
    torch::Tensor wrap_phase(torch::Tensor phase) {
        return torch::remainder(phase + M_PI, 2.0 * M_PI) - M_PI;
    }

// Helper for 1D linear interpolation.
// data: tensor of shape (Time, Feats)
// indices: float tensor of shape (NewTime) with indices to sample at.
    torch::Tensor linear_interp1d(const torch::Tensor& data, const torch::Tensor& indices) {
        long time_dim = data.size(0);
        auto indices_floor = torch::floor(indices).to(torch::kLong);
        auto indices_ceil = torch::ceil(indices).to(torch::kLong);
        auto alpha = (indices - indices_floor).unsqueeze(1);

        // Clamp indices to be within bounds
        indices_floor = torch::clamp(indices_floor, 0, time_dim - 1);
        indices_ceil = torch::clamp(indices_ceil, 0, time_dim - 1);

        auto data_floor = data.index_select(0, indices_floor);
        auto data_ceil = data.index_select(0, indices_ceil);

        return (1.0 - alpha) * data_floor + alpha * data_ceil;
    }

} // namespace

namespace xt::transforms::signal {

    SpeedPerturbation::SpeedPerturbation(
            int sample_rate,
            double min_speed,
            double max_speed,
            int n_fft,
            int hop_length,
            int win_length,
            double p)
            : sample_rate_(sample_rate),
              min_speed_(min_speed),
              max_speed_(max_speed),
              n_fft_(n_fft),
              win_length_(win_length > 0 ? win_length : n_fft),
              hop_length_(hop_length),
              p_(p) {

        if (min_speed_ <= 0 || max_speed_ <= 0 || min_speed_ > max_speed_) {
            throw std::invalid_argument("Invalid speed range.");
        }

        window_ = torch::hann_window(win_length_);
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto SpeedPerturbation::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("SpeedPerturbation::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);
        auto device = waveform.device();

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_) {
            return waveform;
        }

        std::uniform_real_distribution<double> speed_dist(min_speed_, max_speed_);
        double rate = speed_dist(random_engine_);

        if (std::abs(rate - 1.0) < 1e-5) {
            return waveform; // No change
        }

        // --- 2. STFT and Phase Calculation ---
        torch::Tensor stft_out = torch::stft(
                waveform, n_fft_, hop_length_, win_length_, window_.to(device),
                false, true, true
        );
        long num_bins = stft_out.size(0);
        long num_frames = stft_out.size(1);

        auto magnitude = torch::abs(stft_out);
        auto phase = torch::angle(stft_out);

        // Calculate the phase advance between frames
        auto phase_advance = phase.slice(1, 1) - phase.slice(1, 0, -1);

        // Expected phase advance for stationary sinusoids
        auto freq_hz = torch::linspace(0, sample_rate_ / 2.0, num_bins, device);
        auto expected_phase_advance = (2.0 * M_PI * freq_hz * hop_length_) / sample_rate_;

        // Calculate instantaneous frequency (deviation from expected advance)
        auto inst_freq = expected_phase_advance.unsqueeze(1) + wrap_phase(phase_advance - expected_phase_advance.unsqueeze(1));
        inst_freq = torch::cat({inst_freq.slice(1, 0, 1), inst_freq}, 1); // Pad first frame

        // --- 3. Time-domain Resampling of Spectrogram Frames ---
        long new_num_frames = static_cast<long>(std::round(num_frames / rate));
        auto time_indices = torch::linspace(0, num_frames - 1, new_num_frames, device);

        // Interpolate magnitude and instantaneous frequency at the new time points
        auto new_magnitude = linear_interp1d(magnitude.t(), time_indices).t();
        auto new_inst_freq = linear_interp1d(inst_freq.t(), time_indices).t();

        // --- 4. Reconstruct Phase ---
        auto new_phase = torch::empty_like(new_magnitude);
        // Start with the phase of the first frame
        new_phase.select(1, 0).copy_(phase.select(1, 0));

        // Loop to accumulate phase based on the new instantaneous frequencies
        for (long t = 1; t < new_num_frames; ++t) {
            new_phase.select(1, t).copy_(
                    new_phase.select(1, t - 1) + new_inst_freq.select(1, t)
            );
        }

        // --- 5. Inverse STFT ---
        auto new_stft = torch::polar(new_magnitude, new_phase);
        long new_length = static_cast<long>(std::round(waveform.size(-1) / rate));

        torch::Tensor stretched_waveform = torch::istft(
                new_stft, n_fft_, hop_length_, win_length_, window_.to(device),
                true, false, new_length
        );

        return stretched_waveform;
    }

} // namespace xt::transforms::signal