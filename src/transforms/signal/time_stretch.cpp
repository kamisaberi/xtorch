#include "include/transforms/signal/time_stretch.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load an audio file.
    auto [waveform, sample_rate] = xt::utils::audio::load("some_audio_file.wav");
    std::cout << "Original waveform length: " << waveform.size(0) << std::endl;

    // 2. Create a transform to make the audio 50% longer (slower).
    xt::transforms::signal::TimeStretch stretcher_slow(1.5, sample_rate);
    torch::Tensor slow_waveform = std::any_cast<torch::Tensor>(
        stretcher_slow.forward({waveform})
    );
    std::cout << "Slowed (1.5x) waveform length: " << slow_waveform.size(0) << std::endl;

    // 3. Create a transform to make the audio 20% shorter (faster).
    xt::transforms::signal::TimeStretch stretcher_fast(0.8, sample_rate);
    torch::Tensor fast_waveform = std::any_cast<torch::Tensor>(
        stretcher_fast.forward({waveform})
    );
    std::cout << "Sped up (0.8x) waveform length: " << fast_waveform.size(0) << std::endl;

    // 4. Save the results.
    xt::utils::audio::save("stretched_slow.wav", slow_waveform, sample_rate);
    xt::utils::audio::save("stretched_fast.wav", fast_waveform, sample_rate);

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

        indices_floor = torch::clamp(indices_floor, 0, time_dim - 1);
        indices_ceil = torch::clamp(indices_ceil, 0, time_dim - 1);

        auto data_floor = data.index_select(0, indices_floor);
        auto data_ceil = data.index_select(0, indices_ceil);

        return (1.0 - alpha) * data_floor + alpha * data_ceil;
    }

} // namespace

namespace xt::transforms::signal {

    TimeStretch::TimeStretch(
            double fixed_rate,
            int sample_rate,
            int n_fft,
            int hop_length,
            int win_length)
            : fixed_rate_(fixed_rate),
              sample_rate_(sample_rate),
              n_fft_(n_fft),
              win_length_(win_length > 0 ? win_length : n_fft),
              hop_length_(hop_length) {

        if (fixed_rate_ <= 0) {
            throw std::invalid_argument("fixed_rate must be positive.");
        }

        window_ = torch::hann_window(win_length_);
    }

    auto TimeStretch::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("TimeStretch::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);
        auto device = waveform.device();

        // If rate is 1.0, no change is needed.
        if (std::abs(fixed_rate_ - 1.0) < 1e-5) {
            return waveform;
        }

        // Convert stretch factor to internal speed rate
        double rate = 1.0 / fixed_rate_;

        // --- 2. STFT and Phase Calculation ---
        torch::Tensor stft_out = torch::stft(
                waveform, n_fft_, hop_length_, win_length_, window_.to(device),
                false, true, true
        );
        long num_bins = stft_out.size(0);
        long num_frames = stft_out.size(1);

        auto magnitude = torch::abs(stft_out);
        auto phase = torch::angle(stft_out);

        auto phase_advance = phase.slice(1, 1) - phase.slice(1, 0, -1);

        auto freq_hz = torch::linspace(0, sample_rate_ / 2.0, num_bins, device);
        auto expected_phase_advance = (2.0 * M_PI * freq_hz * hop_length_) / sample_rate_;

        auto inst_freq = expected_phase_advance.unsqueeze(1) + wrap_phase(phase_advance - expected_phase_advance.unsqueeze(1));
        inst_freq = torch::cat({inst_freq.slice(1, 0, 1), inst_freq}, 1);

        // --- 3. Time-domain Resampling of Spectrogram Frames ---
        long new_num_frames = static_cast<long>(std::round(num_frames * fixed_rate_));
        auto time_indices = torch::linspace(0, num_frames - 1, new_num_frames, device);

        auto new_magnitude = linear_interp1d(magnitude.t(), time_indices).t();
        auto new_inst_freq = linear_interp1d(inst_freq.t(), time_indices).t();

        // --- 4. Reconstruct Phase ---
        auto new_phase = torch::empty_like(new_magnitude);
        new_phase.select(1, 0).copy_(phase.select(1, 0));

        for (long t = 1; t < new_num_frames; ++t) {
            new_phase.select(1, t).copy_(
                    new_phase.select(1, t - 1) + new_inst_freq.select(1, t)
            );
        }

        // --- 5. Inverse STFT ---
        auto new_stft = torch::polar(new_magnitude, new_phase);
        long new_length = static_cast<long>(std::round(waveform.size(-1) * fixed_rate_));

        torch::Tensor stretched_waveform = torch::istft(
                new_stft, n_fft_, hop_length_, win_length_, window_.to(device),
                true, false, new_length
        );

        return stretched_waveform;
    }

} // namespace xt::transforms::signal