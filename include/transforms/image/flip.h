#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class Flip
     * @brief An image transformation that flips an image horizontally, vertically,
     *        or both.
     *
     * This is a very common and effective data augmentation technique. This
     * implementation uses OpenCV's flip function for the underlying computation.
     */
    class Flip : public xt::Module {
    public:
        /**
         * @brief Default constructor. Defaults to a horizontal flip.
         */
        Flip();

        /**
         * @brief Constructs the Flip transform with a specific flip mode.
         * @param mode A string specifying the flip direction.
         *             Can be "horizontal", "vertical", or "both".
         */
        explicit Flip(const std::string& mode);

        /**
         * @brief Executes the flip operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting flipped torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // The integer flip code used by OpenCV:
        //  1: Horizontal flip (around y-axis)
        //  0: Vertical flip (around x-axis)
        // -1: Both horizontal and vertical flip
        int flip_code_;
    };

} // namespace xt::transforms::image