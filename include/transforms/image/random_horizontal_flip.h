#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomHorizontalFlip
     * @brief Flips an image randomly horizontally with a given probability.
     */
    class RandomHorizontalFlip : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates a horizontal flip transform with a 50% probability.
         */
        RandomHorizontalFlip();

        /**
         * @brief Constructs the RandomHorizontalFlip transform.
         *
         * @param p The probability of the flip being applied. Must be in [0, 1].
         */
        explicit RandomHorizontalFlip(double p);

        /**
         * @brief Executes the random horizontal flip operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting flipped torch::Tensor.
         *         The image may be unchanged if the probability check fails.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image