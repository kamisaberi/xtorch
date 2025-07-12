#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomOrder
     * @brief A meta-transform that applies a list of given transforms in a random order.
     *
     * This transform takes a collection of other transformation modules and, for
     * each forward pass, shuffles their order before applying them sequentially.
     * This increases the variability of the augmentation pipeline.
     */
    class RandomOrder : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an empty transform that does nothing.
         */
        RandomOrder();

        /**
         * @brief Constructs the RandomOrder transform.
         *
         * @param transforms A vector of shared pointers to other transform modules.
         *                   The ownership of these modules is shared with this class.
         */
        explicit RandomOrder(
            const std::vector<std::shared_ptr<xt::Module>>& transforms
        );

        /**
         * @brief Executes the sequence of transforms in a random order.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W), which will be passed to the first
         *                transform in the shuffled sequence.
         * @return An std::any containing the resulting torch::Tensor after all
         *         transforms have been applied.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<std::shared_ptr<xt::Module>> transforms_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image