/**
 * @file a_m_lines.h
 * @brief Declaration of the A-M Lines activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
/**
     * @brief Computes the A-M Lines activation function for an input tensor.
     *
     * Applies a multi-segment piecewise linear activation function defined by a negative slope,
     * a threshold, and a high positive slope.
     *
     * @param x Input tensor to apply the activation function on.
     * @param negative_slope Slope for values less than zero (default: 0.01).
     * @param threshold Positive threshold value dividing the linear segments (default: 1.0).
     * @param high_positive_slope Slope for values exceeding the threshold (default: 0.5).
     * @return torch::Tensor Output tensor with the A-M Lines activation applied.
     */
torch::Tensor am_lines(
    const torch::Tensor& x,
    double negative_slope = 0.01,
    double threshold = 1.0,
    double high_positive_slope = 0.5);

/**
     * @struct AMLines
     * @brief High-level module wrapper for the A-M Lines activation function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
struct AMLines : xt::Module {
public:

    /**
         * @brief Default constructor for AMLines.
         */
    AMLines() = default;

    /**
         * @brief Forward pass for the AMLines module.
         *
         * Expects an input initializer list containing a tensor as its primary argument.
         *
         * @param tensors Initializer list containing inputs (wrapped in `std::any`).
         * @return std::any Output tensor wrapped in `std::any`.
         */
    auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

};
}
