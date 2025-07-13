#pragma once

#include "../common.h"


namespace xt::transforms::signal {

    /**
     * @class MelSpectrogram
     * @brief A transform that creates a Mel-frequency spectrogram from a raw audio waveform.
     *
     * This transform encapsulates the entire process: performing an STFT, calculating
     * the power spectrogram, and then mapping the linear frequencies to the Mel scale
     * by applying a triangular filter bank.
     */
    class MelSpectrogram : public xt::Module {
    public:
        /**
         * @brief Constructs the MelSpectrogram transform.
         *
         * @param sample_rate The sample rate of the audio.
         * @param n_fft The size of the FFT.
         * @param win_length The size of the STFT window. If 0, defaults to `n_fft`.
         * @param hop_length The number of samples between successive STFT frames.
         * @param f_min The minimum frequency for the Mel scale.
         * @param f_max The maximum frequency for the Mel scale. Can be nullopt to use
         *              `sample_rate / 2.0`.
         * @param n_mels The number of Mel bins to generate.
         * @param power The exponent for the magnitude spectrogram (1.0 for magnitude,
         *              2.0 for power). Defaults to 2.0.
         */
        explicit MelSpectrogram(
            int sample_rate = 16000,
            int n_fft = 1024,
            int win_length = 0,
            int hop_length = 256,
            double f_min = 0.0,
            c10::optional<double> f_max = c10::nullopt,
            int n_mels = 128,
            double power = 2.0);

        /**
         * @brief Executes the Mel spectrogram calculation.
         *
         * @param tensors An initializer list expected to contain a single 1D (waveform)
         *                or 2D (batch, waveform) audio tensor.
         * @return An std::any containing the resulting Mel spectrogram tensor of shape
         *         (..., n_mels, time).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Helper function to create the Mel filter banks.
        // This is the same as the one used in InverseMelScale.
        static torch::Tensor create_mel_filter_banks(
            int n_stft,
            int n_mels,
            int sample_rate,
            double f_min,
            double f_max);

        // Pre-computed tensors for efficiency
        torch::Tensor mel_basis_;
        torch::Tensor window_;

        // Stored parameters
        int n_fft_;
        int win_length_;
        int hop_length_;
        double power_;
    };

} // namespace xt::transforms::signal