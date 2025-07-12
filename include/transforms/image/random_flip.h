#pragma once
#include "../common.h"


namespace xt::transforms::image {

    /**
     * @enum FlipOrientation
     * @brief Specifies the orientation for the flip operation.
     */
    enum class FlipOrientation {
        Horizontal,
        Vertical
    };

    /**
     * @class RandomFlip
     * @brief Flips an image randomly either horizontally or vertically with a given probability.
     */
    class RandomFlip : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates a horizontal flip transform with a 50% probability.
         */
        RandomFlip();

        /**
         * @brief Constructs the RandomFlip transform.
         *
         * @param orientation The axis along which to flip the image (Horizontal or Vertical).
         * @param p The probability of the flip being applied. Must be in [0, 1].
         */
        explicit RandomFlip(
            FlipOrientation orientation,
            double p = 0.5
        );

        /**
         * @brief Executes the random flip operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting flipped torch::Tensor.
         *         The image may be unchanged if the probability check fails.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        FlipOrientation orientation_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image