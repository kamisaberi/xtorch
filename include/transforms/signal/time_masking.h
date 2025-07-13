#pragma once

#include "../common.h"



namespace xt::transforms::signal {

    /**
     * @class TimeMasking
     * @brief A signal transformation that applies time masking to a spectrogram.
     *
     * This is a common data augmentation technique from SpecAugment. It randomly
     * selects a range of consecutive time frames in a spectrogram and sets
     * their values to zero. This encourages the model to be more robust to
     * occlusions or losses in short temporal segments of the audio.
     *
     * @note This transform expects a spectrogram as input (e.g., from an STFT),
     *       not a raw waveform. The input tensor shape should be
     *       (..., num_frequency_bins, num_time_frames).
     */
    class TimeMasking : public xt::Module {
    public:
        /**
         * @brief Constructs the TimeMasking transform.
         *
         * @param time_mask_param The maximum number of time frames to mask. A
         *                        random number of frames T from [0, time_mask_param)
         *                        will be chosen.
         * @param num_masks The number of time masks to apply. Defaults to 1.
         * @param p The probability of applying the transform. Defaults to 1.0.
         */
        explicit TimeMasking(
                int time_mask_param,
                int num_masks = 1,
                double p = 1.0);

        /**
         * @brief Executes the time masking operation.
         *
         * @param tensors An initializer list expected to contain a single spectrogram
         *                tensor. This can be a complex or magnitude spectrogram.
         * @return An std::any containing the resulting masked spectrogram tensor
         *         with the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int time_mask_param_;
        int num_masks_;
        double p_;
        mutable std::mt19937 random_engine_;
    };

} // namespace xt::transforms::signal