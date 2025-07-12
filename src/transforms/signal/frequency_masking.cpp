#include "include/transforms/signal/frequency_masking.h"


/*
// Example Usage (goes in a main.cpp or test file)
#include <iostream>

int main() {
    // 1. Create a dummy spectrogram tensor.
    // Shape: (Channel, Frequency, Time). Let's use 1 channel, 80 freq bins, 100 time steps.
    // Let's fill it with ones to see the effect of masking clearly.
    torch::Tensor spectrogram = torch::ones({1, 80, 100});
    std::cout << "Sum of spectrogram before masking: " << spectrogram.sum().item<float>() << std::endl;

    // 2. Create the transform.
    // Mask up to 20 consecutive frequency bins, and apply 2 separate masks.
    xt::transforms::signal::FrequencyMasking masker(20, 2);

    // 3. Apply the transform.
    torch::Tensor masked_spectrogram = std::any_cast<torch::Tensor>(masker.forward({spectrogram}));

    // 4. Verify the output.
    std::cout << "Sum of spectrogram after masking: " << masked_spectrogram.sum().item<float>() << std::endl;
    std::cout << "Shape of masked spectrogram: " << masked_spectrogram.sizes() << std::endl;

    // The "after" sum should be less than the "before" sum.
    // We can also check a slice to see if it's zeroed out.
    // This part is tricky without knowing the random start/width, but we can inspect the tensor.
    // For example, find a row that is all zeros.
    bool mask_found = false;
    for (int i = 0; i < masked_spectrogram.size(1); ++i) {
        if (masked_spectrogram.index({0, i, "..."}).sum().item<float>() == 0.0) {
            mask_found = true;
            std::cout << "Found a masked frequency bin at index: " << i << std::endl;
            break;
        }
    }
    if (!mask_found) {
        std::cout << "No fully masked frequency bin found (mask might be small or did not apply)." << std::endl;
    }

    return 0;
}
*/


namespace xt::transforms::signal {

    FrequencyMasking::FrequencyMasking(int freq_mask_param, int num_masks, double p)
        : freq_mask_param_(freq_mask_param), num_masks_(num_masks), p_(p) {

        if (freq_mask_param_ < 0) {
            throw std::invalid_argument("freq_mask_param cannot be negative.");
        }
        if (num_masks_ < 1) {
            throw std::invalid_argument("num_masks must be at least 1.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        random_engine_.seed(rd());
    }

    auto FrequencyMasking::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation and Probability Check ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("FrequencyMasking::forward received an empty list.");
        }
        torch::Tensor spectrogram = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!spectrogram.defined()) {
            throw std::invalid_argument("Input tensor passed to FrequencyMasking is not defined.");
        }

        std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
        if (prob_dist(random_engine_) > p_) {
            return spectrogram; // Skip applying the transform
        }

        // The frequency dimension is the second to last dimension.
        long num_freq_bins = spectrogram.size(-2);
        if (num_freq_bins == 0) {
            return spectrogram; // Nothing to mask
        }

        // We must clone the tensor to avoid modifying the original input.
        torch::Tensor masked_spectrogram = spectrogram.clone();

        // --- 2. Apply Masks ---
        for (int i = 0; i < num_masks_; ++i) {
            // Choose the width of the mask
            std::uniform_int_distribution<int> width_dist(0, freq_mask_param_);
            int w = width_dist(random_engine_);

            // Don't apply a mask of zero width
            if (w == 0) {
                continue;
            }

            // Choose the starting frequency bin
            // The range ensures the mask [f0, f0 + w) stays within bounds.
            std::uniform_int_distribution<long> start_dist(0, num_freq_bins - w);
            long f0 = start_dist(random_engine_);

            // Apply the mask by setting the slice to zero.
            // The slice is taken along the frequency dimension (-2).
            masked_spectrogram.slice(/*dim=*/-2, /*start=*/f0, /*end=*/f0 + w).zero_();
        }

        return masked_spectrogram;
    }

} // namespace xt::transforms::signal