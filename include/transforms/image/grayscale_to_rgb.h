//TODO SHOULD CHANGE
#pragma once


#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class GrayscaleToRGB
     * @brief An image transformation that converts a single-channel grayscale image
     *        to a 3-channel RGB image.
     *
     * This is achieved by repeating the single grayscale channel three times to
     * create the R, G, and B channels. This is useful for feeding grayscale
     * images into models that expect a 3-channel input.
     */
    class GrayscaleToRGB : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        GrayscaleToRGB();

        /**
         * @brief Executes the grayscale-to-RGB conversion.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor with one channel, i.e., shape [1, H, W].
         * @return An std::any containing the resulting 3-channel torch::Tensor of
         *         shape [3, H, W].
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::image