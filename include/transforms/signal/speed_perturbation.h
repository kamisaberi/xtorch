#pragma once

#include "../common.h"

namespace xt::transforms::signal {

    /**
     * @class SpeedPerturbation
     * @brief A transform that changes the speed of a waveform without changing its pitch.
     *
     * This is a common data augmentation technique that simulates variations in speaking
     * rate. It uses a phase vocoder to stretch or compress the signal in time while
     * preserving the original pitch, which is crucial for preventing distortion.
     */
    class SpeedPerturbation : public xt::Module {
    public:
        /**
         * @brief Constructs the SpeedPerturbation transform.
         *
         * @param sample_rate The sample rate of the audio.
         * @param min_speed The minimum speed factor. Values < 1.0 slow down the audio.
         * @param max_speed The maximum speed factor. Values > 1.0 speed up the audio.
         * @param n_fft The size of the FFT for the phase vocoder.
         * @param hop_length The number of samples between successive STFT frames.
         * @param win_length The size of the STFT window. If 0, defaults to `n_fft`.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit SpeedPerturbation(
                int sample_rate,
                double min_speed = 0.9,
                double max_speed = 1.1,
                int n_fft = 512,
                int hop_length = 128,
                int win_length = 0,
                double p = 1.0);

        /**
         * @brief Executes the speed perturbation operation.
         *
         * @param tensors An initializer list expected to contain a single 1D audio
         *                tensor (waveform).
         * @return An std::any containing the time-stretched waveform as a
         *         torch::Tensor. The length of the tensor will change.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int sample_rate_;
        double min_speed_;
        double max_speed_;
        int n_fft_;
        int win_length_;
        int hop_length_;
        double p_;

        torch::Tensor window_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal