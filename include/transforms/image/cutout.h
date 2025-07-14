//TODO SHOULD CHANGE
#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Cutout
     * @brief Randomly erases one or more rectangular regions from an image.
     *
     * This transform implements the Cutout augmentation. It randomly selects
     * `num_holes` locations in the image and sets a square patch of a given
     * `size` to a constant `fill` value.
     */
    class Cutout : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates one 16x16 hole.
         */
        Cutout();

        /**
         * @brief Constructs the Cutout transform.
         *
         * @param num_holes The number of patches to erase from the image.
         * @param size A pair `{height, width}` for the size of each patch.
         * @param fill The constant value used for the erased patches.
         * @param p The probability of applying the transform. Must be in [0, 1].
         */
        explicit Cutout(
            int num_holes,
            std::pair<int, int> size,
            double fill = 0.0,
            double p = 0.5
        );

        /**
         * @brief Executes the cutout augmentation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with holes.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_holes_;
        std::pair<int, int> size_;
        double fill_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image