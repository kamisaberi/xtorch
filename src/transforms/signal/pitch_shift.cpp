#include "include/transforms/signal/pitch_shift.h"



/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load an audio file.
    auto [waveform, sample_rate] = xt::utils::audio::load("some_audio_file.wav");

    // 2. Create the PitchShift transform. Let's shift the pitch up by 4 semitones.
    xt::transforms::signal::PitchShift pitch_shifter(sample_rate, 4);

    // 3. Apply the transform.
    torch::Tensor shifted_waveform = std::any_cast<torch::Tensor>(
        pitch_shifter.forward({waveform})
    );
    std::cout << "Pitch shifting complete." << std::endl;

    // 4. Create another transform to shift down by 2 semitones.
    xt::transforms::signal::PitchShift pitch_shifter_down(sample_rate, -2);
    torch::Tensor shifted_down_waveform = std::any_cast<torch::Tensor>(
        pitch_shifter_down.forward({waveform})
    );

    // 5. Save the results to listen.
    xt::utils::audio::save("shifted_up_4.wav", shifted_waveform, sample_rate);
    xt::utils::audio::save("shifted_down_2.wav", shifted_down_waveform, sample_rate);

    return 0;
}
*/

namespace xt::transforms::signal {

    // Helper to wrap phase values to the [-PI, PI] range
    inline torch::Tensor wrap_phase(torch::Tensor phase) {
        return torch::remainder(phase + M_PI, 2.0 * M_PI) - M_PI;
    }

    PitchShift::PitchShift(
            int sample_rate,
            int n_steps,
            int n_fft,
            int win_length,
            int hop_length,
            double p)
            : sample_rate_(sample_rate),
              n_steps_(n_steps),
              n_fft_(n_fft),
              win_length_(win_length > 0 ? win_length : n_fft),
              hop_length_(hop_length),
              p_(p) {

        if (sample_rate_ <= 0) {
            throw std::invalid_argument("Sample rate must be positive.");
        }

        window_ = torch::hann_window(win_length_);
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto PitchShift::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("PitchShift::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_ || n_steps_ == 0) {
            return waveform; // Skip transform
        }

        auto device = waveform.device();
        long original_length = waveform.size(-1);

        // --- 2. Calculate Pitch Shift Factor and STFT ---
        double pitch_shift_factor = std::pow(2.0, static_cast<double>(n_steps_) / 12.0);

        torch::Tensor stft_output = torch::stft(
                waveform, n_fft_, hop_length_, win_length_, window_.to(device),
                false, true, true // not normalized, onesided, return complex
        );

        long num_bins = stft_output.size(0);
        long num_frames = stft_output.size(1);

        // --- 3. Phase Vocoder Setup ---
        // Create an empty spectrogram for the output
        auto shifted_stft = torch::zeros_like(stft_output);
        // Phase accumulator for the output spectrogram
        auto phase_accumulator = torch::zeros({num_bins}, stft_output.options());

        // Expected phase advance for each bin (for a stationary signal)
        auto freq_hz = torch::linspace(0, sample_rate_ / 2.0, num_bins, device);
        auto expected_phase_advance = (2.0 * M_PI * freq_hz * hop_length_) / sample_rate_;

        // --- 4. Main Loop: Process each time frame ---
        // This loop is the core of the phase vocoder. It cannot be fully vectorized
        // due to the frame-to-frame dependency of the phase accumulator.
        for (long t = 0; t < num_frames; ++t) {
            auto frame = stft_output.select(1, t); // Current frame

            // Process each frequency bin in the output frame
            for (long f = 0; f < num_bins; ++f) {
                // Determine the corresponding source bin in the input frame
                long source_f_idx = static_cast<long>(std::round(static_cast<double>(f) / pitch_shift_factor));

                if (source_f_idx < num_bins) {
                    // --- Magnitude Transfer ---
                    // The magnitude is simply copied from the source bin.
                    auto mag = torch::abs(frame[source_f_idx]);

                    // --- Phase Calculation ---
                    // 1. Get the phase from the source bin
                    auto source_phase = torch::angle(frame[source_f_idx]);

                    // 2. Calculate true frequency deviation by removing the expected phase advance.
                    auto true_freq_deviation = source_phase - expected_phase_advance[source_f_idx];

                    // 3. Re-calculate the new phase for the target bin.
                    // New Phase = Previous Accumulated Phase + Expected Phase Advance for *target* bin + True Freq Deviation
                    auto new_phase = phase_accumulator[f] + expected_phase_advance[f] + true_freq_deviation;

                    // 4. Update the accumulator for the next frame
                    phase_accumulator[f] = wrap_phase(new_phase);

                    // 5. Create the new complex number and place it in the output
                    shifted_stft.index_put_({f, t}, torch::polar(mag, phase_accumulator[f]));
                }
            }
        }

        // --- 5. Inverse STFT to get the final waveform ---
        torch::Tensor shifted_waveform = torch::istft(
                shifted_stft, n_fft_, hop_length_, win_length_, window_.to(device),
                true, false, original_length
        );

        return shifted_waveform;
    }

} // namespace xt::transforms::signal