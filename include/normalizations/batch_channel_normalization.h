#pragma once

#include "common.h"

namespace xt::norm
{
    struct BatchChannelNorm : xt::Module
    {
    public:
        BatchChannelNorm(int64_t num_features, double eps = 1e-5, double momentum = 0.1, bool affine = true,
                         bool track_running_stats = true);


        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int64_t num_features_;
        double eps_;
        double momentum_;
        bool affine_;
        bool track_running_stats_;

        // Learnable parameters (if affine is true)
        torch::Tensor gamma_; // scale, named "weight" in PyTorch's BatchNorm
        torch::Tensor beta_; // shift, named "bias" in PyTorch's BatchNorm

        // Buffers for running statistics (if track_running_stats is true)
        torch::Tensor running_mean_;
        torch::Tensor running_var_;
        torch::Tensor num_batches_tracked_; // Not strictly needed for forward, but PyTorch BN has it.
    };
}
