#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class Posterize
     * @brief An image transformation that reduces the number of bits for each color channel.
     *
     * This effect quantizes the color space of an image, giving it a flatter,
     * "poster-like" appearance. The strength of the effect is controlled by the
     * number of bits to keep for each channel.
     */
    class Posterize : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses 4 bits, a common value for a noticeable effect.
         */
        Posterize();

        /**
         * @brief Constructs the Posterize transform.
         * @param bits The number of bits to keep for each color channel. Must be
         *             an integer between 1 and 8.
         */
        explicit Posterize(int bits);

        /**
         * @brief Executes the posterization operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting posterized torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int bits_;
    };

} // namespace xt::transforms::image