#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the SmoothStep activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the SmoothStep activation function on an input tensor.
     *
     * SmoothStep applies smooth Hermite interpolation between 0 and 1 for input values bounded
     * between \p edge0 and \p edge1:
     * \f$ t = \text{clamp}\left(\frac{x - \text{edge0}}{\text{edge1} - \text{edge0}}, 0, 1\right) \f$
     * \f$ \text{SmoothStep}(x) = t^2 (3 - 2t) \f$
     *
     * @param x The input tensor.
     * @param edge0 Lower bound edge for interpolation (defaults to 0.0).
     * @param edge1 Upper bound edge for interpolation (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor smooth_step(const torch::Tensor& x, double edge0 = 0.0, double edge1 = 1.0);

    /**
     * @brief A module wrapper for the SmoothStep activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct SmoothStep : xt::Module {
    public:
        /**
         * @brief Default constructor for the SmoothStep module.
         */
        SmoothStep() = default;

        /**
         * @brief Performs the forward pass for the SmoothStep module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}