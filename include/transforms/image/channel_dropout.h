#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class ChannelDropout
     * @brief An image transformation that randomly zeros out entire channels of an image.
     *
     * This is a form of structured dropout that acts as a strong regularizer,
     * forcing the model to not be overly reliant on any single input channel.
     * It is applied independently to each channel.
     */
    class ChannelDropout : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a dropout probability of 0.5.
         */
        ChannelDropout();

        /**
         * @brief Constructs the ChannelDropout transform.
         * @param p The probability of an element (channel) to be zeroed.
         *          Should be in the range [0, 1].
         */
        explicit ChannelDropout(double p);

        /**
         * @brief Executes the channel dropout operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with some channels
         *         potentially zeroed out. The shape and type remain the same.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_; // Probability of dropping a channel
    };

} // namespace xt::transforms::image