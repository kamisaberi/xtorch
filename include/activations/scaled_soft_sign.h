#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the ScaledSoftSign activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the Scaled Softsign activation function on an input tensor.
     *
     * Scaled Softsign applies input and output scaling hyperparameters to the Softsign function:
     * \f$ \text{ScaledSoftSign}(x) = \text{scale\_out} \cdot \frac{\text{scale\_in} \cdot x}{1 + |\text{scale\_in} \cdot x|} \f$
     *
     * @param x The input tensor.
     * @param scale_in Scaling factor applied to the input before activation (defaults to 1.0).
     * @param scale_out Scaling factor applied to the output after activation (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor scaled_soft_sign(const torch::Tensor& x, double scale_in = 1.0, double scale_out = 1.0);

    /**
     * @brief A module wrapper for the ScaledSoftSign activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct ScaledSoftSign : xt::Module {
    public:
        /**
         * @brief Default constructor for the ScaledSoftSign module.
         */
        ScaledSoftSign() = default;

        /**
         * @brief Performs the forward pass for the ScaledSoftSign module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}