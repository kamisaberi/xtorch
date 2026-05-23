#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the SIREN (Sinusoidal Representation Network) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the SIREN sinusoidal activation function on an input tensor.
     *
     * SIREN applies a periodic sinusoidal activation scaled by a frequency hyperparameter \p omega_0:
     * \f$ \text{Siren}(x) = \sin(\omega_0 \cdot x) \f$
     *
     * @param x The input tensor.
     * @param omega_0 Frequency scaling parameter controlling the periodicity (defaults to 30.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor siren(const torch::Tensor& x, double omega_0 = 30.0);

    /**
     * @brief A module wrapper for the SIREN activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct Siren : xt::Module {
    public:
        /**
         * @brief Default constructor for the Siren module.
         */
        Siren() = default;

        /**
         * @brief Performs the forward pass for the Siren module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}