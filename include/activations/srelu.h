#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the SReLU (S-shaped Rectified Linear Unit) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the SReLU (S-shaped Rectified Linear Unit) activation function on an input tensor.
     *
     * SReLU is a piecewise linear activation function defined by learnable/configurable
     * left and right threshold and slope parameters:
     * \f$ \text{SReLU}(x) = \begin{cases}
     * t_{\text{left}} + a_{\text{left}} (x - t_{\text{left}}) & \text{if } x \le t_{\text{left}} \\
     * x & \text{if } t_{\text{left}} < x < t_{\text{right}} \\
     * t_{\text{right}} + a_{\text{right}} (x - t_{\text{right}}) & \text{if } x \ge t_{\text{right}}
     * \end{cases} \f$
     *
     * @param x The input tensor.
     * @param t_left Threshold tensor for the left linear region.
     * @param a_left Slope tensor for the left linear region.
     * @param t_right Threshold tensor for the right linear region.
     * @param a_right Slope tensor for the right linear region.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor srelu(
    const torch::Tensor& x,
    const torch::Tensor& t_left, // Threshold for left part
    const torch::Tensor& a_left, // Slope for left part
    const torch::Tensor& t_right, // Threshold for right part
    const torch::Tensor& a_right // Slope for right part
);


    /**
     * @brief A module wrapper for the SReLU activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct SReLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the SReLU module.
         */
        SReLU() = default;

        /**
         * @brief Performs the forward pass for the SReLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as `x`, `t_left`, `a_left`, `t_right`, and `a_right`.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}