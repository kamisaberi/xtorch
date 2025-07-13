#pragma once

#include "../common.h"

namespace xt::transforms::signal {

    /**
     * @class PitchShift
     * @brief A transform that shifts the pitch of a waveform without changing its duration.
     *
     * This transform implements a phase vocoder to change the pitch. It works in the
     * time-frequency domain by re-mapping frequency bin magnitudes and re-calculating
     * phase information to maintain coherence, which prevents metallic artifacts.
     */
    class PitchShift : public xt::Module {
    public:
        /**
         * @brief Constructs the PitchShift transform.
         *
         * @param sample_rate The sample rate of the audio.
         * @param n_steps The number of semitones (half-steps) to shift the pitch.
         *                Positive values shift the pitch up, negative values shift it down.
         * @param n_fft The size of the FFT.
         * @param win_length The size of the STFT window. If 0, defaults to `n_fft`.
         * @param hop_length The number of samples between successive STFT frames.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit PitchShift(
                int sample_rate,
                int n_steps,
                int n_fft = 1024,
                int win_length = 0,
                int hop_length = 256,
                double p = 1.0);

        /**
         * @brief Executes the pitch shifting operation.
         *
         * @param tensors An initializer list expected to contain a single 1D audio
         *                tensor (waveform).
         * @return An std::any containing the resulting pitch-shifted waveform as a
         *         torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int sample_rate_;
        int n_steps_;
        int n_fft_;
        int win_length_;
        int hop_length_;
        double p_;

        torch::Tensor window_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal