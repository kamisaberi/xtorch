#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class ChannelShuffle
     * @brief An image transformation that shuffles channels between groups.
     *
     * This operation is the core component of the ShuffleNet architecture. It aids
     * information flow across different channel groups in models that use
     * grouped convolutions.
     */
    class ChannelShuffle : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses 2 groups by default.
         */
        ChannelShuffle();

        /**
         * @brief Constructs the ChannelShuffle transform.
         * @param groups The number of groups to divide the channels into for shuffling.
         *               The total number of channels must be divisible by this value.
         */
        explicit ChannelShuffle(int groups);

        /**
         * @brief Executes the channel shuffling operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with shuffled
         *         channels. The shape and type remain the same.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int groups_;
    };

} // namespace xt::transforms::image