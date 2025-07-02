// #include "include/normalizations/sync_bn.h"
//
//
// // #include <torch/torch.h>
// // #include <iostream>
// // #include <vector>
// // #include <numeric> // For std::iota
// //
// // // --- Distributed Communication Placeholders ---
// // // In a real distributed setup, you'd use torch::distributed functions.
// // // These are placeholders to illustrate the logic.
// // namespace dist_ops_placeholder {
// //     // Placeholder for all-reduce. In reality, this would use torch::distributed::all_reduce.
// //     // It would sum tensors from all processes and distribute the sum back.
// //     torch::Tensor all_reduce_sum(const torch::Tensor& tensor) {
// //         // In a single-process setup, this is just the tensor itself.
// //         // In a distributed setup, this is where the magic happens.
// //         if (torch::distributed::is_available() && torch::distributed::is_initialized()) {
// //             // This is a simplified placeholder. Real all_reduce might need a ProcessGroup.
// //             // The actual operation is more complex and depends on the backend.
// //             // For now, we'll just log that it would happen.
// //             // std::cout << "[SyncBN Placeholder] Would perform all_reduce_sum on tensor of shape " << tensor.sizes() << std::endl;
// //             auto world_size = static_cast<double>(torch::distributed::get_world_size());
// //             if (world_size > 1) {
// //                 // If we were to actually do it:
// //                 // torch::Tensor result = tensor.clone();
// //                 // torch::distributed::all_reduce(result, torch::distributed::ReduceOp::SUM);
// //                 // return result;
// //                 // For this placeholder, assume the input tensor is from one process,
// //                 // and we'd need to get data from others.
// //                 // This placeholder will just return the input, effectively making it regular BN.
// //                 // A true implementation would require a full distributed context.
// //                 return tensor * world_size; // SIMULATING the sum from world_size identical tensors
// //             }
// //         }
// //         return tensor;
// //     }
// //
// //     int64_t get_world_size_placeholder() {
// //         if (torch::distributed::is_available() && torch::distributed::is_initialized()) {
// //             return torch::distributed::get_world_size();
// //         }
// //         return 1; // Default to 1 if not distributed
// //     }
// // } // namespace dist_ops_placeholder
// //
// //
// // // Forward declaration for the Impl struct
// // struct SyncBatchNormImpl;
// //
// // // The main module struct that users will interact with.
// // struct SyncBatchNorm : torch::nn::ModuleHolder<SyncBatchNormImpl> {
// //     using torch::nn::ModuleHolder<SyncBatchNormImpl>::ModuleHolder;
// //
// //     torch::Tensor forward(torch::Tensor x) {
// //         return impl_->forward(x);
// //     }
// // };
// //
// // // The implementation struct for SyncBatchNorm
// // struct SyncBatchNormImpl : torch::nn::Module {
// //     int64_t num_features_;
// //     double eps_;
// //     double momentum_;
// //     bool affine_;
// //     bool track_running_stats_; // SyncBN typically tracks running stats
// //
// //     // Learnable parameters (gamma and beta)
// //     torch::Tensor gamma_;
// //     torch::Tensor beta_;
// //
// //     // Buffers for running statistics (global estimates)
// //     torch::Tensor running_mean_;
// //     torch::Tensor running_var_;
// //     torch::Tensor num_batches_tracked_;
// //
// //     SyncBatchNormImpl(int64_t num_features, double eps = 1e-5, double momentum = 0.1,
// //                       bool affine = true, bool track_running_stats = true)
// //         : num_features_(num_features),
// //           eps_(eps),
// //           momentum_(momentum),
// //           affine_(affine),
// //           track_running_stats_(track_running_stats) {
// //         TORCH_CHECK(num_features > 0, "num_features must be positive.");
// //
// //         if (affine_) {
// //             gamma_ = register_parameter("weight", torch::ones({num_features_}));
// //             beta_ = register_parameter("bias", torch::zeros({num_features_}));
// //         }
// //
// //         if (track_running_stats_) {
// //             running_mean_ = register_buffer("running_mean", torch::zeros({num_features_}));
// //             running_var_ = register_buffer("running_var", torch::ones({num_features_}));
// //             num_batches_tracked_ = register_buffer("num_batches_tracked", torch::tensor(0, torch::kLong));
// //         }
// //     }
// //
// //     torch::Tensor forward(torch::Tensor x) {
// //         // Input x: (N_local, C, D1, D2, ...) where C is num_features_
// //         // N_local is the batch size on the current device.
// //
// //         TORCH_CHECK(x.dim() >= 2, "Input tensor must have at least 2 dimensions (N, C, ...). Got ", x.dim());
// //         TORCH_CHECK(x.size(1) == num_features_,
// //                     "Number of input features (channels) mismatch. Expected ", num_features_,
// //                     ", but got ", x.size(1));
// //
// //         // Reshape affine params for broadcasting (e.g., to [1, C, 1, 1] for 4D input)
// //         std::vector<int64_t> affine_param_view_shape(x.dim(), 1);
// //         affine_param_view_shape[1] = num_features_;
// //
// //         torch::Tensor current_global_mean;
// //         torch::Tensor current_global_var;
// //
// //         if (this->is_training()) {
// //             // --- Training mode: Calculate global mean and variance across all devices ---
// //             int64_t N_local = x.size(0);
// //             int64_t num_spatial_elements = 1;
// //             for (int64_t i = 2; i < x.dim(); ++i) {
// //                 num_spatial_elements *= x.size(i);
// //             }
// //
// //             // 1. Calculate sum and sum_sq on the local device for each channel
// //             // Sum over N_local, D1, D2, ... dimensions, keeping channel dimension C.
// //             std::vector<int64_t> reduce_dims_local_stats; // N, D1, D2, ...
// //             reduce_dims_local_stats.push_back(0);
// //             for (int64_t i = 2; i < x.dim(); ++i) {
// //                 reduce_dims_local_stats.push_back(i);
// //             }
// //             // local_sum has shape (C,)
// //             torch::Tensor local_sum = x.sum(reduce_dims_local_stats);
// //             // local_sum_sq has shape (C,)
// //             torch::Tensor local_sum_sq = x.pow(2).sum(reduce_dims_local_stats);
// //
// //             // --- !!! CRUCIAL SYNCHRONIZATION STEP (Placeholder) !!! ---
// //             // In a real SyncBN, these local sums (and N_local * num_spatial_elements)
// //             // would be all-reduced across devices.
// //             torch::Tensor global_sum = dist_ops_placeholder::all_reduce_sum(local_sum);
// //             torch::Tensor global_sum_sq = dist_ops_placeholder::all_reduce_sum(local_sum_sq);
// //             int64_t world_size = dist_ops_placeholder::get_world_size_placeholder();
// //             // Assume all local batches have the same N_local for this placeholder.
// //             // A real implementation gets total count via all-reduce too.
// //             int64_t N_global_count_per_channel = N_local * num_spatial_elements * world_size;
// //             // --- End of Placeholder Synchronization ---
// //
// //             if (N_global_count_per_channel == 0) { // Should not happen with valid inputs
// //                 current_global_mean = torch::zeros({num_features_}, x.options());
// //                 current_global_var = torch::ones({num_features_}, x.options());
// //             } else {
// //                 current_global_mean = global_sum / N_global_count_per_channel;
// //                 // Global Var(X) = E[X^2] - (E[X])^2
// //                 // = (global_sum_sq / N_total) - (global_mean)^2
// //                 current_global_var = (global_sum_sq / N_global_count_per_channel) - current_global_mean.pow(2);
// //             }
// //
// //             // Update running statistics using these global estimates
// //             if (track_running_stats_) {
// //                 running_mean_ = (1.0 - momentum_) * running_mean_ + momentum_ * current_global_mean.detach();
// //                 running_var_  = (1.0 - momentum_) * running_var_  + momentum_ * current_global_var.detach();
// //                 if (num_batches_tracked_) num_batches_tracked_ += 1;
// //             }
// //         } else {
// //             // --- Evaluation mode: Use the saved global running statistics ---
// //             TORCH_CHECK(track_running_stats_, "track_running_stats must be true for SyncBatchNorm in eval mode.");
// //             current_global_mean = running_mean_;
// //             current_global_var = running_var_;
// //         }
// //
// //         // Normalize x using the (global) current_mean and current_var
// //         // These mean/var are (C,), need to be reshaped for broadcasting.
// //         torch::Tensor x_normalized = (x - current_global_mean.view(affine_param_view_shape)) /
// //                                      torch::sqrt(current_global_var.view(affine_param_view_shape) + eps_);
// //
// //         // Apply learnable affine transformation
// //         if (affine_) {
// //             return x_normalized * gamma_.view(affine_param_view_shape) + beta_.view(affine_param_view_shape);
// //         } else {
// //             return x_normalized;
// //         }
// //     }
// //
// //     void pretty_print(std::ostream& stream) const override {
// //         stream << "SyncBatchNorm(num_features=" << num_features_
// //                << ", eps=" << eps_ << ", momentum=" << momentum_
// //                << ", affine=" << (affine_ ? "true" : "false")
// //                << ", track_running_stats=" << (track_running_stats_ ? "true" : "false") << ")";
// //         stream << "\n  (Note: True synchronization requires a distributed environment and backend calls.)";
// //     }
// // };
// // TORCH_MODULE(SyncBatchNorm);
// //
// //
// // // --- Example Usage (will behave like BatchNorm in single-process) ---
// // int main(int argc, char* argv[]) {
// //     torch::manual_seed(0);
// //
// //     // --- Mock Distributed Setup (for testing the logic flow) ---
// //     // In a real application, torch::distributed::init_process_group would be called.
// //     // For this example, we'll just show that if `is_initialized` is false, it defaults to local.
// //     // To simulate, one would need to run this in multiple processes with MPI/Gloo/NCCL.
// //     bool is_distributed_mock = false; // Set to true if you could mock torch::distributed calls
// //     if (is_distributed_mock) {
// //         // This is where you'd init process group.
// //         // E.g., torch::distributed::init_process_group("gloo", "file:///tmp/sharedfile", rank, world_size);
// //         std::cout << "Mocking distributed environment (not actually functional for sync here)." << std::endl;
// //     } else {
// //         std::cout << "Running in single-process mode (SyncBatchNorm will behave like BatchNorm)." << std::endl;
// //     }
// //
// //
// //     int64_t num_features = 3;
// //     SyncBatchNorm sync_bn_module(num_features, /*eps=*/1e-5, /*momentum=*/0.1);
// //     // std::cout << sync_bn_module << std::endl;
// //
// //     // --- Test Case 1: 4D input (like CNN features NCHW) ---
// //     std::cout << "\n--- Test Case 1: 4D input (NCHW) ---" << std::endl;
// //     int64_t N_local = 2, H1 = 5, W1 = 5; // Local batch size
// //     torch::Tensor input1 = torch::randn({N_local, num_features, H1, W1}) * 2 + 3;
// //
// //     std::cout << "Initial running_mean: " << sync_bn_module->running_mean_ << std::endl;
// //     std::cout << "Initial running_var: " << sync_bn_module->running_var_ << std::endl;
// //
// //     // Training pass
// //     sync_bn_module->train();
// //     torch::Tensor output1_train = sync_bn_module->forward(input1);
// //     std::cout << "Output1_train shape: " << output1_train.sizes() << std::endl;
// //     std::cout << "Output1_train [:,0,:,:] mean (local stats, ~0): " << output1_train.select(1,0).mean().item<double>() << std::endl;
// //     std::cout << "Output1_train [:,0,:,:] std (local stats, ~1): " << output1_train.select(1,0).std(false).item<double>() << std::endl;
// //
// //     std::cout << "Updated running_mean (based on local sum * world_size_mock): " << sync_bn_module->running_mean_ << std::endl;
// //     std::cout << "Updated running_var (based on local sum_sq * world_size_mock): " << sync_bn_module->running_var_ << std::endl;
// //
// //
// //     // Evaluation pass (should use running_mean and running_var)
// //     sync_bn_module->eval();
// //     torch::Tensor output1_eval = sync_bn_module->forward(input1);
// //     std::cout << "Output1_eval shape: " << output1_eval.sizes() << std::endl;
// //     std::cout << "Output1_eval [:,0,:,:] mean (uses running stats): " << output1_eval.select(1,0).mean().item<double>() << std::endl;
// //     std::cout << "Output1_eval [:,0,:,:] std (uses running stats): " << output1_eval.select(1,0).std(false).item<double>() << std::endl;
// //
// //     TORCH_CHECK(!torch::allclose(output1_train.select(1,0).mean(), output1_eval.select(1,0).mean(), 1e-5, 1e-5),
// //                 "Train and Eval mode outputs means should differ for SyncBatchNorm.");
// //
// //     // --- Test Case 2: Check backward pass ---
// //     std::cout << "\n--- Test Case 2: Backward pass check ---" << std::endl;
// //     sync_bn_module->train();
// //     torch::Tensor x2 = torch::randn({N_local, num_features, H1, W1}, torch::requires_grad());
// //     torch::Tensor y2 = sync_bn_module->forward(x2);
// //     torch::Tensor loss = y2.mean();
// //     loss.backward();
// //
// //     bool grad_exists_x2 = x2.grad().defined() && x2.grad().abs().sum().item<double>() > 0;
// //     bool grad_exists_gamma = sync_bn_module->gamma_.grad().defined() &&
// //                              sync_bn_module->gamma_.grad().abs().sum().item<double>() > 0;
// //
// //     std::cout << "Gradient exists for x2: " << (grad_exists_x2 ? "true" : "false") << std::endl;
// //     std::cout << "Gradient exists for gamma: " << (grad_exists_gamma ? "true" : "false") << std::endl;
// //     TORCH_CHECK(grad_exists_x2, "No gradient for x2!");
// //     TORCH_CHECK(grad_exists_gamma, "No gradient for gamma!");
// //
// //     std::cout << "\nSyncBatchNorm (logic structure) tests finished." << std::endl;
// //     std::cout << "If run without a true distributed backend, it behaves like standard BatchNorm." << std::endl;
// //
// //     // Clean up distributed resources if they were initialized
// //     // if (torch::distributed::is_available() && torch::distributed::is_initialized()) {
// //     //     torch::distributed::destroy_process_group();
// //     // }
// //     return 0;
// // }
//
//
// namespace xt::norm
// {
//
// namespace dist_ops_placeholder {
//     // Placeholder for all-reduce. In reality, this would use torch::distributed::all_reduce.
//     // It would sum tensors from all processes and distribute the sum back.
//     torch::Tensor all_reduce_sum(const torch::Tensor& tensor) {
//         // In a single-process setup, this is just the tensor itself.
//         // In a distributed setup, this is where the magic happens.
//         if (torch::distributed::is_available() && torch::distributed::is_initialized()) {
//             // This is a simplified placeholder. Real all_reduce might need a ProcessGroup.
//             // The actual operation is more complex and depends on the backend.
//             // For now, we'll just log that it would happen.
//             // std::cout << "[SyncBN Placeholder] Would perform all_reduce_sum on tensor of shape " << tensor.sizes() << std::endl;
//             auto world_size = static_cast<double>(torch::distributed::get_world_size());
//             if (world_size > 1) {
//                 // If we were to actually do it:
//                 // torch::Tensor result = tensor.clone();
//                 // torch::distributed::all_reduce(result, torch::distributed::ReduceOp::SUM);
//                 // return result;
//                 // For this placeholder, assume the input tensor is from one process,
//                 // and we'd need to get data from others.
//                 // This placeholder will just return the input, effectively making it regular BN.
//                 // A true implementation would require a full distributed context.
//                 return tensor * world_size; // SIMULATING the sum from world_size identical tensors
//             }
//         }
//         return tensor;
//     }
//
//     int64_t get_world_size_placeholder() {
//         if (torch::distributed::is_available() && torch::distributed::is_initialized()) {
//             return torch::distributed::get_world_size();
//         }
//         return 1; // Default to 1 if not distributed
//     }
// } // namespace dist_ops_placeholder
//
//
//     SyncBatchNorm::SyncBatchNorm(int64_t num_features, double eps, double momentum,
//                                  bool affine, bool track_running_stats)
//         : num_features_(num_features),
//           eps_(eps),
//           momentum_(momentum),
//           affine_(affine),
//           track_running_stats_(track_running_stats)
//     {
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//
//         if (affine_)
//         {
//             gamma_ = register_parameter("weight", torch::ones({num_features_}));
//             beta_ = register_parameter("bias", torch::zeros({num_features_}));
//         }
//
//         if (track_running_stats_)
//         {
//             running_mean_ = register_buffer("running_mean", torch::zeros({num_features_}));
//             running_var_ = register_buffer("running_var", torch::ones({num_features_}));
//             num_batches_tracked_ = register_buffer("num_batches_tracked", torch::tensor(0, torch::kLong));
//         }
//     }
//
//     auto SyncBatchNorm::forward(std::initializer_list<std::any> tensors) -> std::any
//     {
//         vector<std::any> tensors_ = tensors;
//         auto x = std::any_cast<torch::Tensor>(tensors_[0]);
//
//
//          // Input x: (N_local, C, D1, D2, ...) where C is num_features_
//          // N_local is the batch size on the current device.
//
//          TORCH_CHECK(x.dim() >= 2, "Input tensor must have at least 2 dimensions (N, C, ...). Got ", x.dim());
//          TORCH_CHECK(x.size(1) == num_features_,
//                      "Number of input features (channels) mismatch. Expected ", num_features_,
//                      ", but got ", x.size(1));
//
//          // Reshape affine params for broadcasting (e.g., to [1, C, 1, 1] for 4D input)
//          std::vector<int64_t> affine_param_view_shape(x.dim(), 1);
//          affine_param_view_shape[1] = num_features_;
//
//          torch::Tensor current_global_mean;
//          torch::Tensor current_global_var;
//
//          if (this->is_training()) {
//              // --- Training mode: Calculate global mean and variance across all devices ---
//              int64_t N_local = x.size(0);
//              int64_t num_spatial_elements = 1;
//              for (int64_t i = 2; i < x.dim(); ++i) {
//                  num_spatial_elements *= x.size(i);
//              }
//
//              // 1. Calculate sum and sum_sq on the local device for each channel
//              // Sum over N_local, D1, D2, ... dimensions, keeping channel dimension C.
//              std::vector<int64_t> reduce_dims_local_stats; // N, D1, D2, ...
//              reduce_dims_local_stats.push_back(0);
//              for (int64_t i = 2; i < x.dim(); ++i) {
//                  reduce_dims_local_stats.push_back(i);
//              }
//              // local_sum has shape (C,)
//              torch::Tensor local_sum = x.sum(reduce_dims_local_stats);
//              // local_sum_sq has shape (C,)
//              torch::Tensor local_sum_sq = x.pow(2).sum(reduce_dims_local_stats);
//
//              // --- !!! CRUCIAL SYNCHRONIZATION STEP (Placeholder) !!! ---
//              // In a real SyncBN, these local sums (and N_local * num_spatial_elements)
//              // would be all-reduced across devices.
//              torch::Tensor global_sum = dist_ops_placeholder::all_reduce_sum(local_sum);
//              torch::Tensor global_sum_sq = dist_ops_placeholder::all_reduce_sum(local_sum_sq);
//              int64_t world_size = dist_ops_placeholder::get_world_size_placeholder();
//              // Assume all local batches have the same N_local for this placeholder.
//              // A real implementation gets total count via all-reduce too.
//              int64_t N_global_count_per_channel = N_local * num_spatial_elements * world_size;
//              // --- End of Placeholder Synchronization ---
//
//              if (N_global_count_per_channel == 0) { // Should not happen with valid inputs
//                  current_global_mean = torch::zeros({num_features_}, x.options());
//                  current_global_var = torch::ones({num_features_}, x.options());
//              } else {
//                  current_global_mean = global_sum / N_global_count_per_channel;
//                  // Global Var(X) = E[X^2] - (E[X])^2
//                  // = (global_sum_sq / N_total) - (global_mean)^2
//                  current_global_var = (global_sum_sq / N_global_count_per_channel) - current_global_mean.pow(2);
//              }
//
//              // Update running statistics using these global estimates
//              if (track_running_stats_) {
//                  running_mean_ = (1.0 - momentum_) * running_mean_ + momentum_ * current_global_mean.detach();
//                  running_var_  = (1.0 - momentum_) * running_var_  + momentum_ * current_global_var.detach();
//                  if (num_batches_tracked_) num_batches_tracked_ += 1;
//              }
//          } else {
//              // --- Evaluation mode: Use the saved global running statistics ---
//              TORCH_CHECK(track_running_stats_, "track_running_stats must be true for SyncBatchNorm in eval mode.");
//              current_global_mean = running_mean_;
//              current_global_var = running_var_;
//          }
//
//          // Normalize x using the (global) current_mean and current_var
//          // These mean/var are (C,), need to be reshaped for broadcasting.
//          torch::Tensor x_normalized = (x - current_global_mean.view(affine_param_view_shape)) /
//                                       torch::sqrt(current_global_var.view(affine_param_view_shape) + eps_);
//
//          // Apply learnable affine transformation
//          if (affine_) {
//              return x_normalized * gamma_.view(affine_param_view_shape) + beta_.view(affine_param_view_shape);
//          } else {
//              return x_normalized;
//          }
//
//
//
//     }
// }
