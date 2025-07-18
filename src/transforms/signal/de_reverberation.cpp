#include <transforms/signal/de_reverberation.h>


/*
// Example Usage (goes in a main.cpp or test file)
#include "include/utils/audio/io.h" // Assumes the audio I/O utility exists
#include <iostream>

int main() {
    // 1. Load a reverberant audio file
    // You would need a real audio file for this to work.
    // Let's create a dummy one for demonstration.
    // A clean signal
    torch::Tensor clean_signal = torch::sin(2 * M_PI * 440.0 * torch::arange(0, 16000) / 16000.0);
    // A simple exponential decay impulse response to simulate reverb
    torch::Tensor impulse_response = torch::exp(-torch::arange(0, 8000) / 1000.0) * torch::randn({8000});
    impulse_response /= torch::sum(torch::abs(impulse_response));
    // Convolve to create a reverberant signal
    torch::Tensor reverberant_signal = at::conv1d(
        clean_signal.view({1, 1, -1}), impulse_response.view({1, 1, -1}),
        {}, 1, "same"
    ).squeeze();

    std::cout << "Original (reverberant) signal length: " << reverberant_signal.size(0) << std::endl;
    // xt::utils::audio::save("reverberant.wav", reverberant_signal, 16000);

    // 2. Create the transform
    // Use fairly aggressive settings for a noticeable effect
    xt::transforms::signal::DeReverberation dereverb_transform(0.9, 80.0);

    // 3. Apply the transform
    torch::Tensor dereverberated_signal = std::any_cast<torch::Tensor>(
        dereverb_transform.forward({reverberant_signal})
    );

    // 4. Verify the output
    std::cout << "Dereverberated signal length: " << dereverberated_signal.size(0) << std::endl;

    // Save the output to listen to the difference
    // xt::utils::audio::save("dereverberated.wav", dereverberated_signal, 16000);
    // xt::utils::audio::save("clean_original.wav", clean_signal, 16000);

    return 0;
}
*/


namespace xt::transforms::signal {

    DeReverberation::DeReverberation(
        double suppression_level,
        double reverb_time_constant_ms,
        double p)
        : suppression_level_(suppression_level),
          reverb_time_constant_ms_(reverb_time_constant_ms),
          p_(p) {

        if (suppression_level_ < 0.0 || suppression_level_ > 1.0) {
            throw std::invalid_argument("suppression_level must be between 0.0 and 1.0.");
        }
        if (reverb_time_constant_ms_ <= 0.0) {
            throw std::invalid_argument("reverb_time_constant_ms must be positive.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto DeReverberation::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("DeReverberation::forward received an empty list.");
        }
        torch::Tensor signal = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!signal.defined()) {
            throw std::invalid_argument("Input tensor passed to DeReverberation is not defined.");
        }
        if (signal.dim() != 1) {
            // This algorithm is simpler for 1D signals. Multi-channel would require iterating.
            throw std::invalid_argument("DeReverberation currently only supports 1D (mono) tensors.");
        }

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_) {
            return signal; // Skip applying the transform
        }

        long original_length = signal.size(0);
        auto device = signal.device();
        auto dtype = signal.dtype();

        // --- 2. STFT: Transform to Time-Frequency Domain ---
        torch::Tensor window = torch::hann_window(win_length_, device);
        torch::Tensor stft_output = torch::stft(
            signal,
            n_fft_,
            hop_length_,
            win_length_,
            window,
            /*normalized=*/false,
            /*onesided=*/true,
            /*return_complex=*/true
        );

        torch::Tensor magnitude = torch::abs(stft_output);

        // --- 3. Estimate Reverberation Power ---
        // This is the core heuristic. We estimate the reverb magnitude by applying a
        // time-domain IIR filter (low-pass filter) to the magnitude spectrogram.
        // This smooths out the sharp transients of the direct signal, leaving an
        // estimate of the slower-decaying reverb.
        torch::Tensor reverb_mag_estimate = torch::empty_like(magnitude);

        // Calculate decay factor from the time constant
        // Assumes a sample rate of 16000 Hz, a common value for speech.
        // For a more robust solution, sample rate should be an input.
        double sample_rate = 16000.0;
        double hop_seconds = static_cast<double>(hop_length_) / sample_rate;
        double time_constant_seconds = reverb_time_constant_ms_ / 1000.0;
        double decay_factor = std::exp(-hop_seconds / time_constant_seconds);

        // Initialize the first frame of the estimate
        reverb_mag_estimate.select(1, 0).copy_(magnitude.select(1, 0));

        // Iterate through time frames to apply the IIR filter
        for (long t = 1; t < magnitude.size(1); ++t) {
            // reverb[t] = decay * reverb[t-1] + (1 - decay) * mag[t]
            // This is a simple first-order low-pass filter
            torch::Tensor smoothed = decay_factor * reverb_mag_estimate.select(1, t - 1) +
                                     (1.0 - decay_factor) * magnitude.select(1, t);
            reverb_mag_estimate.select(1, t).copy_(smoothed);
        }

        // --- 4. Create and Apply Suppression Mask ---
        // The mask is calculated as: max(0, 1 - (reverb_est / magnitude))
        auto eps = torch::tensor(1e-8, torch::kFloat32).to(device);
        torch::Tensor mask = torch::clamp_min(
            1.0 - suppression_level_ * (reverb_mag_estimate / (magnitude + eps)),
            0.0
        );

        // Apply the mask to the complex spectrogram
        torch::Tensor dereverberated_stft = stft_output * mask;

        // --- 5. ISTFT: Transform back to Time Domain ---
        torch::Tensor dereverberated_signal = torch::istft(
            dereverberated_stft,
            n_fft_,
            hop_length_,
            win_length_,
            window,
            /*onesided=*/true,
            /*normalized=*/false,
            /*length=*/original_length
        );

        return dereverberated_signal.to(dtype);
    }

} // namespace xt::transforms::signal