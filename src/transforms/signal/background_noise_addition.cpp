#include <transforms/signal/background_noise_addition.h>

// #include "include/transforms/signal/background_noise_addition.h"

// This is an *assumed* utility header for loading audio.
// You will need to implement this part, for example, using a library like libsndfile or dr_wav.
#include "include/utils/audio.h"

#include <stdexcept>
#include <cmath>

/*
// Example Usage (goes in a main.cpp or test file)
// Note: This requires a dummy audio loading function and a dummy noise file.
// For a real test, you'd use a proper audio library.

#include "utils/audio/io.h" // Your real audio I/O header
#include <iostream>

// Dummy implementation for demonstration
namespace xt::utils::audio {
    // This function would normally load an audio file from disk
    auto load(const std::string& path) -> std::pair<torch::Tensor, int> {
        std::cout << "--- (Dummy) Loading audio from: " << path << " ---" << std::endl;
        // Let's pretend we loaded a 1-second sine wave at 16kHz
        if (path == "clean_signal.wav") {
             return {torch::sin(2 * M_PI * 440.0 * torch::arange(0, 16000) / 16000.0), 16000};
        }
        // Let's pretend we loaded 2 seconds of random noise at 16kHz
        else {
             return {torch::randn({32000}), 16000};
        }
    }
}

int main() {
    // 1. Create a dummy clean signal tensor
    auto [signal, sr] = xt::utils::audio::load("clean_signal.wav");
    std::cout << "Original signal length: " << signal.size(0) << std::endl;

    // 2. Define noise files and create the transform
    std::vector<std::string> noise_files = {"noise1.wav", "noise2.wav"};
    // Add noise with an SNR between 5 and 15 dB
    xt::transforms::signal::BackgroundNoiseAddition noise_adder(noise_files, 5.0, 15.0);

    // 3. Apply the transform
    torch::Tensor noisy_signal = std::any_cast<torch::Tensor>(noise_adder.forward({signal}));

    // 4. Verify the output
    std::cout << "Noisy signal length: " << noisy_signal.size(0) << std::endl;

    // The noisy signal should have the same length as the original
    // but its values will be different.
    // To verify, you could calculate the power of the original vs the added noise.
    torch::Tensor added_noise = noisy_signal - signal;
    float signal_power = torch::mean(torch::pow(signal, 2)).item<float>();
    float noise_power = torch::mean(torch::pow(added_noise, 2)).item<float>();
    float snr_db = 10 * std::log10(signal_power / noise_power);

    std::cout << "Signal Power: " << signal_power << std::endl;
    std::cout << "Added Noise Power: " << noise_power << std::endl;
    std::cout << "Resulting SNR (dB): " << snr_db << " (should be between 5 and 15)" << std::endl;

    return 0;
}
*/

namespace xt::transforms::signal {

    BackgroundNoiseAddition::BackgroundNoiseAddition(
        const std::vector<std::string>& noise_paths,
        double snr_min,
        double snr_max,
        double p)
        : noise_paths_(noise_paths), snr_min_(snr_min), snr_max_(snr_max), p_(p) {

        if (noise_paths_.empty()) {
            throw std::invalid_argument("noise_paths vector cannot be empty.");
        }
        if (snr_min_ > snr_max_) {
            throw std::invalid_argument("snr_min cannot be greater than snr_max.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        // Seed the random number generator
        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto BackgroundNoiseAddition::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("BackgroundNoiseAddition::forward received an empty list.");
        }
        torch::Tensor signal = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!signal.defined()) {
            throw std::invalid_argument("Input tensor passed to BackgroundNoiseAddition is not defined.");
        }

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_) {
            return signal; // Skip applying the transform
        }

        // --- 2. Load and Prepare Noise ---
        // Randomly select a noise file
        std::uniform_int_distribution<size_t> noise_dist(0, noise_paths_.size() - 1);
        const std::string& noise_path = noise_paths_[noise_dist(random_engine_)];

        // This is the assumed utility function to load audio. It should return
        // the audio data as a tensor and the sample rate.
        auto [noise, noise_sr] = xt::utils::audio::load(noise_path);
        noise = noise.to(signal.options()); // Ensure noise tensor has same dtype/device

        // --- 3. Align Signal and Noise Lengths ---
        long signal_len = signal.size(-1);
        long noise_len = noise.size(-1);

        if (noise_len < signal_len) {
            // Tile the noise if it's shorter than the signal
            long n_repeats = (signal_len + noise_len - 1) / noise_len;
            noise = noise.repeat({n_repeats});
            noise = noise.slice(0, 0, signal_len);
        } else if (noise_len > signal_len) {
            // Randomly crop the noise if it's longer
            std::uniform_int_distribution<long> start_dist(0, noise_len - signal_len);
            long start_index = start_dist(random_engine_);
            noise = noise.slice(0, start_index, start_index + signal_len);
        }
        // If lengths are equal, do nothing.

        // --- 4. Calculate and Apply SNR Scaling ---
        // Select a random SNR value from the specified range
        std::uniform_real_distribution<double> snr_dist(snr_min_, snr_max_);
        double target_snr_db = snr_dist(random_engine_);

        // Calculate power of the signal and noise
        // Power is the mean of squares. Add a small epsilon to avoid division by zero.
        auto eps = torch::tensor(1e-10, signal.options());
        auto signal_power = torch::mean(torch::pow(signal, 2));
        auto noise_power = torch::mean(torch::pow(noise, 2));

        // Calculate the required scaling factor for the noise
        double snr_linear = std::pow(10.0, target_snr_db / 10.0);
        auto scale = torch::sqrt(signal_power / (snr_linear * noise_power + eps));

        // --- 5. Add Scaled Noise to Signal ---
        torch::Tensor noisy_signal = signal + noise * scale;

        return noisy_signal;
    }

} // namespace xt::transforms::signal