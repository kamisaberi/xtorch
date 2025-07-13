#pragma once

#include "../common.h"

#include <random>        // For random number generation

namespace xt::transforms::video {

    /**
     * @class RandomClipReverse
     * @brief A video data augmentation that randomly reverses the frames of a clip.
     *
     * With a given probability, this transform will reverse the temporal order of
     * the frames in a video clip. This can help models learn that an action is
     * the same, regardless of the playback direction.
     */
    class RandomClipReverse : public xt::Module {
    public:
        /**
         * @brief Constructs the RandomClipReverse transform.
         *
         * @param reverse_prob The probability (from 0.0 to 1.0) that the clip
         *                     will be reversed. Defaults to 0.5.
         */
        explicit RandomClipReverse(float reverse_prob = 0.5f);

        /**
         * @brief Executes the random reversal operation.
         * @param tensors An initializer list expected to contain a single video
         *                clip as a torch::Tensor of shape (T, C, H, W), where T
         *                is the temporal/frame dimension.
         * @return An std::any containing the (potentially) reversed clip as a
         *         torch::Tensor with the same shape as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float reverse_prob_;
        std::mt19937 random_engine_;
        std::uniform_real_distribution<float> prob_distribution_;
    };

} // namespace xt::transforms::video