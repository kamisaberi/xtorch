#pragma once

#include "../common.h"


namespace xt::transforms::video { // A new namespace for video-specific transforms

    /**
     * @class TemporalJitter
     * @brief A video transformation that applies a random temporal shift to a sequence of frames.
     *
     * This data augmentation technique improves a model's robustness to variations
     * in the timing of actions within a video clip. It works by randomly shifting
     * the start and end points of the input sequence.
     */
    class TemporalJitter : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a moderate jitter limit.
         */
        TemporalJitter();

        /**
         * @brief Constructs the TemporalJitter transform.
         * @param max_jitter_frames The maximum number of frames to shift the sequence by.
         *                          A random shift between [-max, +max] will be applied.
         */
        explicit TemporalJitter(int max_jitter_frames);

        /**
         * @brief Executes the temporal jitter operation.
         * @param tensors An initializer list expected to contain a single 4D video
         *                tensor of shape [T, C, H, W].
         * @return An std::any containing the resulting temporally-shifted torch::Tensor
         *         with the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int max_jitter_frames_;
    };

} // namespace xt::transforms::video