#pragma once

#include "../common.h"


namespace xt::transforms::signal {

    /**
     * @class BackgroundNoiseAddition
     * @brief A signal transformation that adds background noise to an audio tensor.
     *
     * This transform is a common data augmentation technique for audio. It randomly
     * selects a noise file, adjusts its power to a target Signal-to-Noise Ratio (SNR),
     * and adds it to the input signal.
     */
    class BackgroundNoiseAddition : public xt::Module {
    public:
        /**
         * @brief Constructs the BackgroundNoiseAddition transform.
         * @param noise_paths A vector of file paths to the noise audio files.
         * @param snr_min The minimum Signal-to-Noise Ratio (SNR) in dB.
         * @param snr_max The maximum Signal-to-Noise Ratio (SNR) in dB.
         * @param p The probability of applying the transform. Defaults to 1.0 (always apply).
         */
        explicit BackgroundNoiseAddition(
            const std::vector<std::string>& noise_paths,
            double snr_min,
            double snr_max,
            double p = 1.0);

        /**
         * @brief Executes the noise addition operation.
         * @param tensors An initializer list expected to contain a single 1D or 2D audio
         *                tensor (Samples or Channels x Samples).
         * @return An std::any containing the resulting noisy torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<std::string> noise_paths_;
        double snr_min_;
        double snr_max_;
        double p_;
        // Use mutable to allow modification in the const-like forward method
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal