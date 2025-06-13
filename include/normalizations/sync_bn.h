// #pragma once
//
// #include "common.h"
//
// namespace xt::norm
// {
//     struct SyncBatchNorm : xt::Module
//     {
//     public:
//         SyncBatchNorm(int64_t num_features, double eps = 1e-5, double momentum = 0.1,
//                           bool affine = true, bool track_running_stats = true);
//
//         auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//
//     private:
//         int64_t num_features_;
//         double eps_;
//         double momentum_;
//         bool affine_;
//         bool track_running_stats_; // SyncBN typically tracks running stats
//
//         // Learnable parameters (gamma and beta)
//         torch::Tensor gamma_;
//         torch::Tensor beta_;
//
//         // Buffers for running statistics (global estimates)
//         torch::Tensor running_mean_;
//         torch::Tensor running_var_;
//         torch::Tensor num_batches_tracked_;
//
//     };
// }
