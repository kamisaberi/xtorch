#pragma once

#include "common.h"

namespace xt::norm
{
    struct ModeNorm : xt::Module
    {
    public:
        ModeNorm(int64_t num_features, int64_t num_modes, double eps_in = 1e-5);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_; // Number of features in input x (channels)
        int64_t num_modes_; // Number of different normalization "modes"
        double eps_in_; // Epsilon for the base Instance Normalization

        // Learnable affine parameters (gamma and beta) for each mode.
        // We'll store these as a single tensor and select them, or as separate parameters.
        // Using a single tensor for embedding-like lookup is common.
        torch::Tensor gammas_; // Shape (num_modes_, num_features_)
        torch::Tensor betas_; // Shape (num_modes_, num_features_)

        // Base normalization (Instance Normalization for this example)
        // No learnable affine params for the base IN, as they are mode-specific.

    };
}
