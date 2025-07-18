#include <transforms/signal/griffin_lim.h>




/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h" // Assumes the audio I/O utility exists
#include <iostream>

int main() {
    // 1. Create a dummy magnitude spectrogram.
    // This would typically be the output of a TTS model.
    // For this example, let's create one from a real audio file.
    auto [waveform, sr] = xt::utils::audio::load("some_audio_file.wav");

    int n_fft = 1024;
    int hop_length = 256;
    int win_length = 1024;
    double power = 2.0;

    torch::Tensor window = torch::hann_window(win_length);
    torch::Tensor stft_out = torch::stft(waveform, n_fft, hop_length, win_length, window, false, true, true);
    torch::Tensor magnitude = torch::abs(stft_out);
    torch::Tensor power_spectrogram = torch::pow(magnitude, power);

    std::cout << "Original Spectrogram Shape: " << power_spectrogram.sizes() << std::endl;

    // 2. Create the GriffinLim transform with matching parameters.
    xt::transforms::signal::GriffinLim griffin_lim(30, n_fft, hop_length, win_length, power);

    // 3. Apply the transform to reconstruct the waveform.
    torch::Tensor reconstructed_waveform = std::any_cast<torch::Tensor>(griffin_lim.forward({power_spectrogram}));

    // 4. Verify the output and save it.
    std::cout << "Reconstructed Waveform Length: " << reconstructed_waveform.size(0) << std::endl;
    xt::utils::audio::save("reconstructed_audio.wav", reconstructed_waveform, sr);

    return 0;
}
*/

namespace xt::transforms::signal {

    GriffinLim::GriffinLim(
        int n_iters,
        int n_fft,
        int hop_length,
        int win_length,
        double power,
        double momentum)
        : n_iters_(n_iters),
          n_fft_(n_fft),
          hop_length_(hop_length),
          win_length_(win_length),
          power_(power),
          momentum_(momentum) {

        if (n_iters_ <= 0) {
            throw std::invalid_argument("n_iters must be positive.");
        }
        if (momentum_ < 0.0 || momentum_ >= 1.0) {
            throw std::invalid_argument("Momentum must be in the range [0.0, 1.0).");
        }
    }

    auto GriffinLim::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GriffinLim::forward received an empty list.");
        }
        torch::Tensor mag_spec = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!mag_spec.defined()) {
            throw std::invalid_argument("Input spectrogram passed to GriffinLim is not defined.");
        }

        // --- 2. Preparation ---
        // Clone the input to avoid modification and move to the correct device/type.
        mag_spec = mag_spec.clone();
        auto device = mag_spec.device();

        // If the input is a power spectrogram, convert it to a magnitude spectrogram.
        if (power_ > 1.0) {
            mag_spec = torch::pow(mag_spec, 1.0 / power_);
        }

        // Create the window tensor needed for STFT/ISTFT.
        torch::Tensor window = torch::hann_window(win_length_, torch::kFloat32).to(device);

        // --- 3. Initialize Phase and Momentum ---
        // Start with random phase.
        torch::Tensor angles = (2 * M_PI * torch::rand_like(mag_spec) - M_PI).to(device);
        torch::Tensor phase_update = torch::zeros_like(mag_spec);

        // --- 4. Iterative Phase Estimation ---
        for (int i = 0; i < n_iters_; ++i) {
            // Combine the input magnitude with the current phase estimate.
            torch::Tensor complex_spec = torch::polar(mag_spec, angles);

            // Inverse STFT to get a time-domain signal.
            torch::Tensor waveform = torch::istft(
                complex_spec, n_fft_, hop_length_, win_length_, window, true, false
            );

            // STFT the result back to the frequency domain. This gives us a new
            // spectrogram with a phase that is consistent with the signal.
            torch::Tensor rebuilt_spec = torch::stft(
                waveform, n_fft_, hop_length_, win_length_, window, false, true, true
            );

            // Get the phase from this new spectrogram.
            torch::Tensor rebuilt_angles = torch::angle(rebuilt_spec);

            // Update phase using momentum.
            if (i > 0 && momentum_ > 0.0) {
                // The update is the difference from the previous phase, scaled by momentum.
                phase_update = rebuilt_angles - angles;
                angles = rebuilt_angles + momentum_ * phase_update;
            } else {
                angles = rebuilt_angles;
            }
        }

        // --- 5. Final Synthesis ---
        // Combine the final phase with the original magnitude one last time.
        torch::Tensor final_complex_spec = torch::polar(mag_spec, angles);

        // Synthesize the final waveform.
        torch::Tensor final_waveform = torch::istft(
            final_complex_spec, n_fft_, hop_length_, win_length_, window, true, false
        );

        return final_waveform;
    }

} // namespace xt::transforms::signal