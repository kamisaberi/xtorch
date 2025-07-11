#include "include/transforms/signal/add_noise.h"


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