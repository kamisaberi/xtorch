#pragma once

#include "../common.h"


namespace xt::transforms::signal {

    /**
     * @class Spectrogram
     * @brief A transform that creates a spectrogram from a raw audio waveform.
     *
     * This transform computes the Short-Time Fourier Transform (STFT) of a signal
     * to get its time-frequency representation. The output can be either a magnitude
     * or a power spectrogram.
     */
    class Spectrogram : public xt::Module {
    public:
        /**
         * @brief Constructs the Spectrogram transform.
         *
         * @param n_fft The size of the FFT.
         * @param win_length The size of the STFT window. If 0, defaults to `n_fft`.
         * @param hop_length The number of samples between successive STFT frames.
         * @param power The exponent for the magnitude spectrogram. Use 1.0 for a
         *              magnitude spectrogram or 2.0 for a power spectrogram (default).
         */
        explicit Spectrogram(
                int n_fft = 1024,
                int win_length = 0,
                int hop_length = 256,
                double power = 2.0);

        /**
         * @brief Executes the spectrogram calculation.
         *
         * @param tensors An initializer list expected to contain a single 1D (waveform)
         *                or 2D (batch, waveform) audio tensor.
         * @return An std::any containing the resulting spectrogram tensor of shape
         *         (..., num_frequency_bins, time).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Pre-computed window tensor for efficiency
        torch::Tensor window_;

        // Stored parameters
        int n_fft_;
        int win_length_;
        int hop_length_;
        double power_;
    };

} // namespace xt::transforms::signal