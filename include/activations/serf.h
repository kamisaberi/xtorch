#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the SERF (Log-Softplus ERror Activation Function) and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the SERF (Log-Softplus ERror Activation Function) on an input tensor.
     *
     * SERF is a smooth, self-regularized, non-monotonic activation function defined as:
     * \f$ \text{SERF}(x) = \lambda \cdot x \cdot \text{erf}(\ln(1 + e^{k \cdot x})) \f$
     *
     * @param x The input tensor.
     * @param k_param Scaling factor parameter inside the softplus term (defaults to 2.0).
     * @param lambda_param Output scaling parameter (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor serf(const torch::Tensor& x, double k_param = 2.0, double lambda_param = 1.0);

    /**
     * @brief A module wrapper for the SERF activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct Serf : xt::Module {
    public:
        /**
         * @brief Default constructor for the Serf module.
         */
        Serf() = default;

        /**
         * @brief Performs the forward pass for the Serf module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}