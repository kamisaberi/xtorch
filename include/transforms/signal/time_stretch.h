#pragma once

#include "../common.h"


namespace xt::transforms::signal {

    /**
     * @class TimeStretch
     * @brief A transform that stretches or compresses the duration of a waveform
     *        without changing its pitch.
     *
     * This transform uses a phase vocoder to perform pitch-preserving time scaling.
     * It is a deterministic version of SpeedPerturbation, useful for audio effects
     * or conforming audio to a fixed length.
     */
    class TimeStretch : public xt::Module {
    public:
        /**
         * @brief Constructs the TimeStretch transform.
         *
         * @param fixed_rate The factor by which to stretch the audio.
         *                   - `fixed_rate > 1.0` makes the audio longer (slower).
         *                   - `fixed_rate < 1.0` makes the audio shorter (faster).
         *                   - `fixed_rate = 1.0` has no effect.
         * @param sample_rate The sample rate of the audio.
         * @param n_fft The size of the FFT for the phase vocoder.
         * @param hop_length The number of samples between successive STFT frames.
         * @param win_length The size of the STFT window. If 0, defaults to `n_fft`.
         */
        explicit TimeStretch(
                double fixed_rate,
                int sample_rate,
                int n_fft = 512,
                int hop_length = 128,
                int win_length = 0);

        /**
         * @brief Executes the time stretching operation.
         *
         * @param tensors An initializer list expected to contain a single 1D audio
         *                tensor (waveform).
         * @return An std::any containing the time-stretched waveform as a
         *         torch::Tensor. The length of the tensor will change.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double fixed_rate_;
        int sample_rate_;
        int n_fft_;
        int win_length_;
        int hop_length_;

        torch::Tensor window_;
    };

} // namespace xt::transforms::signal