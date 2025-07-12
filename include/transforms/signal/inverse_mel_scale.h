#pragma once

#include "../common.h"



namespace xt::transforms::signal {

    /**
     * @class InverseMelScale
     * @brief A transform to convert a Mel-frequency spectrogram to a linear-frequency spectrogram.
     *
     * This operation approximates the inverse of a Mel-frequency filter bank. It is
     * essential for models that generate audio in the Mel-spectrogram domain. The resulting
     * linear power spectrogram can then be passed to an algorithm like Griffin-Lim to
     * synthesize a waveform.
     *
     * @note This is an approximation, as the original Mel scaling is a lossy,
     *       non-invertible operation.
     */
    class InverseMelScale : public xt::Module {
    public:
        /**
         * @brief Constructs the InverseMelScale transform.
         *
         * The parameters must match those used to create the original Mel spectrogram.
         *
         * @param n_stft The number of frequency bins in the target linear spectrogram
         *               (typically `n_fft / 2 + 1`).
         * @param n_mels The number of Mel bins in the input spectrogram.
         * @param sample_rate The sample rate of the audio.
         * @param f_min The minimum frequency for the Mel scale.
         * @param f_max The maximum frequency for the Mel scale. Can be nullopt to use
         *              `sample_rate / 2.0`.
         */
        explicit InverseMelScale(
            int n_stft,
            int n_mels = 128,
            int sample_rate = 16000,
            double f_min = 0.0,
            c10::optional<double> f_max = c10::nullopt);

        /**
         * @brief Executes the inverse Mel scaling operation.
         *
         * @param tensors An initializer list expected to contain a single Mel spectrogram
         *                tensor of shape (..., n_mels, time).
         * @return An std::any containing the resulting linear power spectrogram as a
         *         torch::Tensor of shape (..., n_stft, time).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Helper function to create the Mel filter banks.
        static torch::Tensor create_mel_filter_banks(
            int n_stft,
            int n_mels,
            int sample_rate,
            double f_min,
            double f_max);

        // Pre-computed pseudo-inverse of the Mel filter bank.
        torch::Tensor inverse_mel_basis_;
    };

} // namespace xt::transforms::signal