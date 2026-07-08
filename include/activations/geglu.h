#pragma once

#include "common.h"

/**
 * @file
 * @brief Defines the GeGLU (Gated Linear Unit with GELU) activation function and module.
 */

namespace xt::activations {

    /**
     * @brief Applies the GeGLU (Gated Linear Unit with Gaussian Error Linear Unit) activation function.
     *
     * GeGLU splits the input tensor into two equal parts along the specified dimension,
     * applies the GELU activation to the first half, and multiplies it element-wise by the second half:
     * \f$ \text{GeGLU}(x_1, x_2) = \text{GELU}(x_1) \otimes x_2 \f$
     *
     * @param x The input tensor to process.
     * @param dim The dimension along which to split the input tensor (defaults to 1).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor geglu(torch::Tensor x, int64_t dim = 1);

    /**
     * @brief Module wrapper for the GeGLU activation function.
     *
     * Inherits from `xt::Module` to support dynamic invocation via standard module pipelines.
     */
    struct GeGLU: xt::Module {
    public:
        /**
         * @brief Default constructor for the GeGLU module.
         */
        GeGLU() = default;

        /**
         * @brief Performs the forward pass for the GeGLU module.
         *
         * @param tensors An initializer list of `std::any` arguments, expected to contain
         *                the input tensor and optional parameters.
         * @return std::any The result of the GeGLU operation wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}