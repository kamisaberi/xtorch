#include "include/transforms/signal/mel_spectrogram.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h"
#include <iostream>

int main() {
    // 1. Load an audio file.
    auto [waveform, sample_rate] = xt::utils::audio::load("some_audio_file.wav");
    std::cout << "Loaded waveform with shape: " << waveform.sizes() << std::endl;

    // 2. Define parameters. These should be consistent across your project.
    int n_fft = 1024;
    int hop_length = 256;
    int win_length = 1024;
    int n_mels = 80;

    // 3. Create the MelSpectrogram transform.
    xt::transforms::signal::MelSpectrogram mel_spec_transform(
        sample_rate, n_fft, win_length, hop_length, 0.0, c10::nullopt, n_mels
    );

    // 4. Apply the transform to get the Mel spectrogram.
    torch::Tensor mel_spectrogram = std::any_cast<torch::Tensor>(
        mel_spec_transform.forward({waveform})
    );

    // 5. Verify the output shape.
    // Shape should be (n_mels, num_frames), e.g., (80, ...)
    std::cout << "Resulting Mel Spectrogram shape: " << mel_spectrogram.sizes() << std::endl;

    // This mel_spectrogram is now ready to be fed into a neural network.
    // To reconstruct it, you would use InverseMelScale and then GriffinLim.

    return 0;
}
*/

namespace xt::transforms::signal {

    // This is the same robust, vectorized function from InverseMelScale.
    // In a real project, this should be moved to a shared utility file.
    torch::Tensor MelSpectrogram::create_mel_filter_banks(
        int n_stft, int n_mels, int sample_rate, double f_min, double f_max) {

        auto hz_to_mel = [](double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); };

        double min_mel = hz_to_mel(f_min);
        double max_mel = hz_to_mel(f_max);
        auto mel_points = torch::linspace(min_mel, max_mel, n_mels + 2);

        auto hz_points = 700.0 * (torch::pow(10.0, mel_points / 2595.0) - 1.0);

        int n_fft = (n_stft - 1) * 2;
        auto fft_bins = torch::floor((n_fft + 1) * hz_points / sample_rate);

        auto f_start = fft_bins.slice(0, 0, n_mels).unsqueeze(1);
        auto f_center = fft_bins.slice(0, 1, n_mels + 1).unsqueeze(1);
        auto f_end = fft_bins.slice(0, 2, n_mels + 2).unsqueeze(1);
        auto all_freqs = torch::arange(0, n_stft).unsqueeze(0);

        auto left_ramp = (all_freqs - f_start) / (f_center - f_start);
        auto right_ramp = (f_end - all_freqs) / (f_end - f_center);

        auto mel_basis = torch::max(
            torch::zeros({1}),
            torch::min(left_ramp, right_ramp)
        );

        return mel_basis;
    }

    MelSpectrogram::MelSpectrogram(
        int sample_rate,
        int n_fft,
        int win_length,
        int hop_length,
        double f_min,
        c10::optional<double> f_max_opt,
        int n_mels,
        double power)
        : n_fft_(n_fft),
          win_length_(win_length > 0 ? win_length : n_fft),
          hop_length_(hop_length),
          power_(power) {

        // Pre-compute the STFT window.
        window_ = torch::hann_window(win_length_);

        // Pre-compute the Mel filter bank matrix.
        int n_stft = n_fft / 2 + 1;
        double f_max = f_max_opt.value_or(static_cast<double>(sample_rate) / 2.0);
        mel_basis_ = create_mel_filter_banks(n_stft, n_mels, sample_rate, f_min, f_max);
    }


    auto MelSpectrogram::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("MelSpectrogram::forward received an empty list.");
        }
        torch::Tensor waveform = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!waveform.defined()) {
            throw std::invalid_argument("Input tensor passed to MelSpectrogram is not defined.");
        }

        auto device = waveform.device();

        // --- 2. Perform STFT ---
        // `torch::stft` returns a complex tensor.
        torch::Tensor stft_output = torch::stft(
            waveform,
            n_fft_,
            hop_length_,
            win_length_,
            window_.to(device),
            /*normalized=*/false,
            /*onesided=*/true,
            /*return_complex=*/true
        );

        // --- 3. Calculate Power Spectrogram ---
        // Take the magnitude of the complex STFT output.
        torch::Tensor power_spectrogram = torch::abs(stft_output);
        // Raise to the specified power (e.g., 2.0 for power, 1.0 for magnitude).
        power_spectrogram = torch::pow(power_spectrogram, power_);

        // --- 4. Apply Mel Filter Bank ---
        // Matrix multiplication maps the linear spectrogram to the Mel scale.
        // Shapes: (n_mels, n_stft) @ (..., n_stft, time) -> (..., n_mels, time)
        torch::Tensor mel_spectrogram = torch::matmul(
            mel_basis_.to(device),
            power_spectrogram
        );

        return mel_spectrogram;
    }

} // namespace xt::transforms::signal