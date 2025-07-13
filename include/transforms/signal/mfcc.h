#pragma once

#include "../common.h"

#include "mel_spectrogram.h" // MFCC is built on top of MelSpectrogram

namespace xt::transforms::signal {

    /**
     * @class MFCC
     * @brief A transform to create Mel-Frequency Cepstral Coefficients (MFCCs) from a waveform.
     *
     * This transform computes a Mel spectrogram, takes its logarithm, and then applies
     * a Discrete Cosine Transform (DCT) to generate the final coefficients. MFCCs are
     * a standard feature for speech recognition and other audio classification tasks.
     */
    class MFCC : public xt::Module {
    public:
        /**
         * @brief Constructs the MFCC transform.
         *
         * @param sample_rate Sample rate of the audio.
         * @param n_mfcc The number of MFCCs to return.
         * @param n_fft The size of the FFT.
         * @param win_length The size of the STFT window.
         * @param hop_length The number of samples between successive STFT frames.
         * @param f_min Minimum frequency for the Mel scale.
         * @param f_max Maximum frequency for the Mel scale.
         * @param n_mels The number of Mel bins to generate.
         * @param log_mels Whether to apply log to the Mel spectrogram. Defaults to true.
         */
        explicit MFCC(
            int sample_rate = 16000,
            int n_mfcc = 40,
            int n_fft = 1024,
            int win_length = 0,
            int hop_length = 256,
            double f_min = 0.0,
            c10::optional<double> f_max = c10::nullopt,
            int n_mels = 128,
            bool log_mels = true);

        /**
         * @brief Executes the MFCC calculation.
         *
         * @param tensors An initializer list expected to contain a single 1D (waveform)
         *                or 2D (batch, waveform) audio tensor.
         * @return An std::any containing the resulting MFCC tensor of shape
         *         (..., n_mfcc, time).
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Helper function to create the DCT matrix.
        static torch::Tensor create_dct_matrix(int n_mels, int n_mfcc);

        // This transform composes a MelSpectrogram transform for the first stage.
        MelSpectrogram mel_spectrogram_;

        // Pre-computed DCT matrix for efficiency.
        torch::Tensor dct_matrix_;

        bool log_mels_;
        const double log_offset_ = 1e-6; // Epsilon for numerical stability in log
    };

} // namespace xt::transforms::signal