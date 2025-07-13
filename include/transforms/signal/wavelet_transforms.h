#pragma once

#include "../common.h"

namespace xt::transforms::signal {

    /**
     * @class WaveletTransform
     * @brief A transform that computes the multi-level 1D Discrete Wavelet Transform (DWT)
     *        of a signal.
     *
     * This transform uses the Mallat algorithm, applying a cascade of filter banks
     * (low-pass and high-pass quadrature mirror filters) to decompose the signal
     * into approximation and detail coefficients at various levels.
     *
     * @note The output is a vector of tensors, not a single tensor.
     */
    class WaveletTransform : public xt::Module {
    public:
        /**
         * @brief Constructs the WaveletTransform.
         *
         * @param wavelet The name of the wavelet to use (e.g., "haar", "db2", "db4").
         * @param n_levels The number of decomposition levels to perform. If 0, the
         *                 maximum possible number of levels will be computed.
         * @param padding_mode The padding mode for convolution at signal boundaries.
         *                     Supported modes: "zeros", "reflect", "replicate", "circular".
         */
        explicit WaveletTransform(
                const std::string& wavelet = "db4",
                int n_levels = 0,
                const std::string& padding_mode = "reflect");

        /**
         * @brief Executes the DWT decomposition.
         *
         * @param tensors An initializer list expected to contain a single 1D audio
         *                tensor (waveform).
         * @return An std::any containing a `std::vector<torch::Tensor>`. The vector
         *         contains the wavelet coefficients in the order [cA_n, cD_n, cD_{n-1}, ..., cD_1],
         *         where cA is the final approximation and cD are the details.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Helper to get wavelet filter coefficients.
        static std::vector<float> get_wavelet_coeffs(const std::string& name);

        // Pre-computed filter tensors.
        torch::Tensor dec_lo_; // Low-pass decomposition filter
        torch::Tensor dec_hi_; // High-pass decomposition filter

        int n_levels_;
        std::string padding_mode_;
    };

} // namespace xt::transforms::signal