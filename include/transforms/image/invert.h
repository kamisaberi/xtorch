#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Invert
     * @brief Inverts the colors of an image.
     *
     * This transform creates a "photographic negative" effect by applying the
     * formula `output = 1.0 - input` to each pixel. This is a deterministic
     * operation that is always applied.
     * The operation is defined for float tensors in the [0, 1] range.
     */
    class Invert : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        Invert();

        /**
         * @brief Executes the color inversion.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with float values in the range [0, 1].
         * @return An std::any containing the resulting inverted torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::image