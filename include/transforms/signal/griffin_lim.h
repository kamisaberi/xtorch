#pragma once

#include "../common.h"



namespace xt::transforms::signal {

    /**
     * @class GriffinLim
     * @brief A signal transformation that synthesizes a waveform from a magnitude spectrogram.
     *
     * This transform implements the Griffin-Lim algorithm, an iterative method for
     * estimating the phase of a signal given only its magnitude spectrogram. It is
     * commonly used to convert the output of audio generation models (like Tacotron)
     * back into a listenable waveform.
     */
    class GriffinLim : public xt::Module {
    public:
        /**
         * @brief Constructs the GriffinLim transform.
         *
         * @param n_iters The number of iterations to run the algorithm. More iterations
         *                generally lead to higher quality but take longer.
         * @param n_fft The size of the FFT, which determines the number of frequency bins.
         *              Must match the STFT settings used to create the spectrogram.
         * @param hop_length The number of samples between successive STFT frames.
         * @param win_length The size of the STFT window. Typically equals `n_fft`.
         * @param power The exponent for the magnitude spectrogram. Use 2.0 for a power
         *              spectrogram (mag**2) or 1.0 for a magnitude spectrogram.
         * @param momentum A factor for accelerating convergence. A value between 0 and 1.
         *                 Set to 0 to disable. A value like 0.99 is common.
         */
        explicit GriffinLim(
            int n_iters = 30,
            int n_fft = 1024,
            int hop_length = 256,
            int win_length = 1024,
            double power = 2.0,
            double momentum = 0.99);

        /**
         * @brief Executes the Griffin-Lim phase reconstruction algorithm.
         *
         * @param tensors An initializer list expected to contain a single magnitude
         *                spectrogram tensor of shape (..., num_frequency_bins, num_time_frames).
         * @return An std::any containing the resulting synthesized audio waveform as a
         *         1D or 2D torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int n_iters_;
        int n_fft_;
        int hop_length_;
        int win_length_;
        double power_;
        double momentum_;
    };

} // namespace xt::transforms::signal