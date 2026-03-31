#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Nipuna activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the Nipuna activation function on an input tensor.
     *
     * Applies the parameterized Nipuna activation transformation to the input tensor \p x
     * using shape/scaling parameters \p a and \p b.
     *
     * @param x The input tensor.
     * @param a Scaling or shape hyperparameter (defaults to 0.25).
     * @param b Secondary scaling or shift hyperparameter (defaults to 0.05).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor nipuna(const torch::Tensor& x, double a = 0.25, double b = 0.05);

    /**
     * @brief A module wrapper for the Nipuna activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network models.
     */
    struct Nipuna : xt::Module {
    public:
        /**
         * @brief Default constructor for the Nipuna module.
         */
        Nipuna() = default;

        /**
         * @brief Performs the forward pass for the Nipuna module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}