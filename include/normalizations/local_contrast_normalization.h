#pragma once

#include "common.h"

namespace xt::norm
{
    struct LocalContrastNorm : xt::Module
    {
    public:
        explicit LocalContrastNorm(int64_t kernel_size = 5,
                                             double alpha = 1.0, // Often 1.0 or related to kernel_size
                                             double beta = 0.5, // Or a small value like 0.0001
                                             double eps = 1e-5);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t kernel_size_; // Size of the local neighborhood window (e.g., 3, 5, 7)
        double alpha_; // Scaling factor for the denominator (contrast term)
        double beta_; // Additive constant in the denominator (regularization)
        double eps_; // Small epsilon for sqrt to prevent NaN if variance is zero

        // Pooling layers for local statistics
        // For local mean: Average pooling
        torch::nn::AvgPool2d local_mean_pool_{nullptr};
        // For local variance/std dev: we need sum of squares and sum of values.
        // Can use AvgPool2d for sum_x / N and sum_x_sq / N.
        // Var(X) = E[X^2] - (E[X])^2
        torch::nn::AvgPool2d local_sq_mean_pool_{nullptr}; // For E[X^2]
    };
}
