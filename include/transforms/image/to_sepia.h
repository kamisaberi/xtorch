#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class ToSepia
     * @brief An image transformation that applies a sepia toning effect.
     *
     * This filter gives an image a warm, brownish monochrome appearance, similar
     * to old photographs. It is achieved by applying a specific linear
     * transformation to the RGB color channels.
     */
    class ToSepia : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        ToSepia();

        /**
         * @brief Executes the sepia toning operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor with 3 channels, i.e., shape [3, H, W].
         * @return An std::any containing the resulting sepia-toned torch::Tensor
         *         with the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // The sepia transformation matrix will be a member for efficiency
        torch::Tensor sepia_matrix_;
    };

} // namespace xt::transforms::image