#include <transforms/signal/add_noise.h>


//
// #include "transforms/signal/add_noise.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy signal tensor (e.g., a clean sine wave)
//     torch::Tensor t = torch::linspace(0, 4 * M_PI, 1000);
//     torch::Tensor signal = torch::sin(t);
//
//     // Calculate the Power/Standard Deviation of the clean signal
//     float original_power = signal.std().item<float>();
//     std::cout << "Original signal power (std dev): " << original_power << std::endl;
//
//     // 2. Instantiate the transform to add noise with a random amplitude
//     //    between 0.1 and 0.3.
//     xt::transforms::signal::AddNoise noiser(0.1f, 0.3f, /*p=*/1.0f);
//
//     // 3. Apply the transform
//     std::any result_any = noiser.forward({signal});
//     torch::Tensor noisy_signal = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     float noisy_power = noisy_signal.std().item<float>();
//     std::cout << "Noisy signal power (std dev): " << noisy_power << std::endl;
//     // The power of the noisy signal should be higher than the original.
//
//     std::cout << "\nOriginal signal shape: " << signal.sizes() << std::endl;
//     std::cout << "Noisy signal shape:    " << noisy_signal.sizes() << std::endl;
//
//     // In a real application, you could plot the original and noisy signals
//     // to visualize the effect.
//
//     return 0;
// }

namespace xt::transforms::signal {

    AddNoise::AddNoise() : min_amplitude_(0.001f), max_amplitude_(0.01f), p_(0.5f) {}

    AddNoise::AddNoise(float min_amplitude, float max_amplitude, float p)
        : min_amplitude_(min_amplitude), max_amplitude_(max_amplitude), p_(p) {

        if (min_amplitude_ < 0 || max_amplitude_ < 0) {
            throw std::invalid_argument("Noise amplitude must be non-negative.");
        }
        if (min_amplitude_ > max_amplitude_) {
            throw std::invalid_argument("min_amplitude cannot be greater than max_amplitude.");
        }
        if (p_ < 0.0f || p_ > 1.0f) {
            throw std::invalid_argument("Probability `p` must be between 0.0 and 1.0.");
        }
    }

    auto AddNoise::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Decide whether to apply the transform ---
        if (torch::rand({1}).item<float>() > p_) {
            return tensors.begin()[0];
        }

        // --- 2. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("AddNoise::forward received an empty list of tensors.");
        }
        torch::Tensor input_signal = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_signal.defined()) {
            throw std::invalid_argument("Input signal passed to AddNoise is not defined.");
        }

        // --- 3. Generate Gaussian Noise ---
        // Choose a random amplitude for this specific application
        float amplitude = min_amplitude_;
        if (max_amplitude_ > min_amplitude_) {
            amplitude = torch::rand({1}).item<float>() * (max_amplitude_ - min_amplitude_) + min_amplitude_;
        }

        if (amplitude == 0.0f) {
            return input_signal; // No noise to add
        }

        // Create a tensor of random numbers from a standard normal distribution (mean=0, std=1)
        // with the same size, device, and type as the input signal.
        torch::Tensor noise = torch::randn_like(input_signal);

        // --- 4. Add Noise to the Signal ---
        // We only need to scale the noise by the chosen amplitude (standard deviation).
        torch::Tensor noisy_signal = input_signal + noise * amplitude;

        // Note: Unlike images, signals are not always clamped to a specific range like [0, 1].
        // If your signals have a known range (e.g., [-1, 1] for audio), you should add clamping here:
        // noisy_signal = torch::clamp(noisy_signal, -1.0, 1.0);

        return noisy_signal;
    }

} // namespace xt::transforms::signal