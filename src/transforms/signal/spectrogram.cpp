#include "include/transforms/signal/spectrogram.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load an audio file.
    auto [waveform, sample_rate] = xt::utils::audio::load("some_audio_file.wav");
    std::cout << "Loaded waveform with shape: " << waveform.sizes() << std::endl;

    // 2. Define STFT parameters.
    int n_fft = 1024;
    int hop_length = 256;
    int win_length = 1024;

    // 3. Create the Spectrogram transform.
    xt::transforms::signal::Spectrogram spec_transform(n_fft, win_length, hop_length);

    // 4. Apply the transform to get the power spectrogram.
    torch::Tensor spectrogram = std::any_cast<torch::Tensor>(
        spec_transform.forward({waveform})
    );

    // 5. Verify the output shape.
    // Shape should be (num_freq_bins, num_frames).
    // num_freq_bins = n_fft / 2 + 1 = 513
    std::cout << "Resulting Spectrogram shape: " << spectrogram.sizes() << std::endl;

    // This spectrogram can be used for visualization or as a feature input.

    return 0;
}
*/

namespace xt::transforms::signal {

    Spectrogram::Spectrogram(
            int n_fft,
            int win_length,
            int hop_length,
            double power)
            : n_fft_(n_fft),
              win_length_(win_length > 0 ? win_length : n_fft),
              hop_length_(hop_length),
              power_(power) {

        if (power_ <= 0) {
            throw std::invalid_argument("Power must be positive.");
        }

        // Pre-compute the STFT window for efficiency.
        // A Hann window is a standard choice.
        window_ = torch::hann_window(win_length_);
    }

    auto Spectrogram::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Spectrogram::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!waveform.defined()) {
            throw std::invalid_argument("Input tensor passed to Spectrogram is not defined.");
        }

        auto device = waveform.device();

        // --- 2. Perform STFT ---
        // `torch::stft` returns a complex tensor. This is the modern and preferred
        // way to handle STFT in LibTorch.
        torch::Tensor stft_output = torch::stft(
                waveform,
                n_fft_,
                hop_length_,
                win_length_,
                window_.to(device), // Ensure window is on the same device as the waveform
                /*normalized=*/false,
                /*onesided=*/true, // For real-valued input, we only need one side of the spectrum
                /*return_complex=*/true
        );

        // --- 3. Calculate Power or Magnitude Spectrogram ---
        // Take the absolute value (magnitude) of the complex STFT output.
        torch::Tensor spec = torch::abs(stft_output);

        // If power > 1, raise the magnitude to the specified power.
        if (power_ != 1.0) {
            spec = torch::pow(spec, power_);
        }

        return spec;
    }

} // namespace xt::transforms::signal