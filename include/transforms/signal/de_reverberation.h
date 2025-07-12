#pragma once

#include "../common.h"



namespace xt::transforms::signal {

    /**
     * @class DeReverberation
     * @brief A signal transformation that attempts to reduce reverberation.
     *
     * This transform applies a simplified dereverberation algorithm based on
     * spectral subtraction in the time-frequency domain. It estimates the
     * reverberant energy floor and subtracts it from the signal's magnitude
     * spectrogram, which can help improve clarity, especially for speech.
     *
     * Note: This is a heuristic-based method and may introduce artifacts. Its
     * effectiveness depends heavily on the nature of the signal and the reverb.
     */
    class DeReverberation : public xt::Module {
    public:
        /**
         * @brief Constructs the DeReverberation transform.
         *
         * @param suppression_level Controls the aggressiveness of the reverb removal.
         *                          Value between 0.0 (no effect) and 1.0 (strong effect).
         *                          Defaults to 0.8.
         * @param reverb_time_constant_ms The time constant (in milliseconds) used for
         *                                smoothing the reverb estimate. Longer times assume
         *                                a more slowly changing reverb tail. Defaults to 50.0.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit DeReverberation(
            double suppression_level = 0.8,
            double reverb_time_constant_ms = 50.0,
            double p = 1.0);

        /**
         * @brief Executes the dereverberation operation.
         * @param tensors An initializer list expected to contain a single 1D audio
         *                tensor (Samples). Multi-channel audio is not yet supported.
         * @return An std::any containing the resulting dereverberated torch::Tensor
         *         with roughly the same length as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double suppression_level_;
        double reverb_time_constant_ms_;
        double p_;

        // STFT parameters - can be exposed in constructor if more control is needed
        int n_fft_ = 1024;
        int hop_length_ = 256;
        int win_length_ = 1024;

        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal