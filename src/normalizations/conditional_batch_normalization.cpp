#include "include/normalizations/conditional_batch_normalization.h"


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <numeric> // For std::iota
//
// // Forward declaration for the Impl struct
// struct ConditionalBatchNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct ConditionalBatchNormalization : torch::nn::ModuleHolder<ConditionalBatchNormalizationImpl> {
//     using torch::nn::ModuleHolder<ConditionalBatchNormalizationImpl>::ModuleHolder;
//
//     // Forward method takes the main input x and the conditioning input
//     torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& conditioning_input) {
//         return impl_->forward(x, conditioning_input);
//     }
// };
//
// // The implementation struct
// struct ConditionalBatchNormalizationImpl : torch::nn::Module {
//     int64_t num_features_;          // Number of features in input x (channels)
//     int64_t cond_embedding_dim_;    // Dimensionality of the conditioning input vector
//     double eps_;
//     double momentum_;
//     bool track_running_stats_;
//     int64_t cond_hidden_dim_;       // Hidden dimension for the conditioning network
//
//     // Buffers for running statistics (unconditional, based on x)
//     torch::Tensor running_mean_;
//     torch::Tensor running_var_;
//     torch::Tensor num_batches_tracked_;
//
//     // Conditioning network layers (to produce gamma and beta)
//     torch::nn::Linear fc_cond1_{nullptr}; // Optional first layer
//     torch::nn::Linear fc_cond_out_{nullptr}; // Output layer for gamma and beta
//
//     ConditionalBatchNormalizationImpl(int64_t num_features,
//                                       int64_t cond_embedding_dim,
//                                       int64_t cond_hidden_dim = 0, // 0 means no hidden layer for cond net
//                                       double eps = 1e-5,
//                                       double momentum = 0.1,
//                                       bool track_running_stats = true)
//         : num_features_(num_features),
//           cond_embedding_dim_(cond_embedding_dim),
//           cond_hidden_dim_(cond_hidden_dim),
//           eps_(eps),
//           momentum_(momentum),
//           track_running_stats_(track_running_stats) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//         TORCH_CHECK(cond_embedding_dim > 0, "cond_embedding_dim must be positive.");
//
//         if (track_running_stats_) {
//             running_mean_ = register_buffer("running_mean", torch::zeros({num_features_}));
//             running_var_ = register_buffer("running_var", torch::ones({num_features_}));
//             num_batches_tracked_ = register_buffer("num_batches_tracked", torch::tensor(0, torch::kLong));
//         }
//
//         // Setup conditioning network
//         if (cond_hidden_dim_ <= 0) { // Direct mapping from conditioning_input to gamma/beta
//             fc_cond_out_ = torch::nn::Linear(cond_embedding_dim_, 2 * num_features_); // 2 for gamma and beta
//             register_module("fc_cond_out", fc_cond_out_);
//         } else {
//             fc_cond1_ = torch::nn::Linear(cond_embedding_dim_, cond_hidden_dim_);
//             fc_cond_out_ = torch::nn::Linear(cond_hidden_dim_, 2 * num_features_);
//             register_module("fc_cond1", fc_cond1_);
//             register_module("fc_cond_out", fc_cond_out_);
//         }
//     }
//
//     torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& conditioning_input) {
//         // x: input tensor (N, C, D1, D2, ...) where C is num_features_
//         // conditioning_input: (N, cond_embedding_dim_)
//
//         TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got ", x.dim());
//         TORCH_CHECK(x.size(0) == conditioning_input.size(0),
//                     "Batch size of x (", x.size(0), ") and conditioning_input (", conditioning_input.size(0), ") must match.");
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Number of input features (channels) in x mismatch. Expected ", num_features_,
//                     ", but got ", x.size(1), " for input x of shape ", x.sizes());
//         TORCH_CHECK(conditioning_input.dim() == 2 && conditioning_input.size(1) == cond_embedding_dim_,
//                     "Conditioning input must be 2D with shape (N, cond_embedding_dim). Got shape: ", conditioning_input.sizes());
//
//         // --- 1. Batch Normalization part (identical to standard BN) ---
//         torch::Tensor current_mean;
//         torch::Tensor current_var;
//
//         std::vector<int64_t> reduce_dims_for_stats; // Dims to average over for mean/var
//         reduce_dims_for_stats.push_back(0); // Batch dimension
//         for (int64_t i = 2; i < x.dim(); ++i) { // Spatial/sequential dimensions
//             reduce_dims_for_stats.push_back(i);
//         }
//
//         if (this->is_training() && track_running_stats_) {
//             torch::Tensor batch_mean = x.mean(reduce_dims_for_stats, /*keepdim=false*/ false);
//             // Variance: E[(X - E[X])^2]
//             auto x_minus_mean_sq = (x - batch_mean.view({1, num_features_, 1, 1})).pow(2); // Reshape for broadcast
//             torch::Tensor batch_var = x_minus_mean_sq.mean(reduce_dims_for_stats, /*keepdim=false*/ false);
//
//             running_mean_ = (1.0 - momentum_) * running_mean_ + momentum_ * batch_mean;
//             running_var_  = (1.0 - momentum_) * running_var_  + momentum_ * batch_var;
//             if (num_batches_tracked_) num_batches_tracked_ += 1;
//
//             current_mean = batch_mean;
//             current_var = batch_var;
//         } else {
//             if (track_running_stats_) {
//                 current_mean = running_mean_;
//                 current_var = running_var_;
//             } else { // No tracking, use current batch stats (less common for eval BN)
//                 current_mean = x.mean(reduce_dims_for_stats, /*keepdim=false*/ false);
//                 auto x_minus_mean_sq = (x - current_mean.view({1, num_features_, 1, 1})).pow(2);
//                 current_var = x_minus_mean_sq.mean(reduce_dims_for_stats, /*keepdim=false*/ false);
//             }
//         }
//
//         // Reshape mean and var to (1, C, 1, 1, ...) for broadcasting
//         std::vector<int64_t> bn_param_view_shape(x.dim(), 1);
//         bn_param_view_shape[1] = num_features_;
//
//         torch::Tensor x_normalized = (x - current_mean.view(bn_param_view_shape)) /
//                                      torch::sqrt(current_var.view(bn_param_view_shape) + eps_);
//
//         // --- 2. Generate Gamma and Beta from conditioning_input ---
//         torch::Tensor cond_features = conditioning_input;
//         if (fc_cond1_) { // If hidden layer exists
//             cond_features = fc_cond1_->forward(cond_features);
//             cond_features = torch::relu(cond_features); // Common activation
//         }
//         torch::Tensor gamma_beta_params = fc_cond_out_->forward(cond_features); // (N, 2 * num_features_)
//
//         auto chunks = torch::chunk(gamma_beta_params, 2, /*dim=*/1);
//         torch::Tensor gamma_generated = chunks[0]; // (N, num_features_)
//         torch::Tensor beta_generated  = chunks[1]; // (N, num_features_)
//
//         // --- 3. Reshape generated Gamma and Beta for broadcasting ---
//         // Desired shape: (N, C, 1, 1, ...) to match x_normalized (N, C, D1, D2, ...)
//         std::vector<int64_t> affine_param_view_shape;
//         affine_param_view_shape.push_back(x.size(0));      // N
//         affine_param_view_shape.push_back(num_features_);  // C
//         for (int64_t i = 2; i < x.dim(); ++i) {
//             affine_param_view_shape.push_back(1);
//         }
//
//         gamma_generated = gamma_generated.view(affine_param_view_shape);
//         beta_generated  = beta_generated.view(affine_param_view_shape);
//
//         // --- 4. Apply conditional affine transformation ---
//         return gamma_generated * x_normalized + beta_generated;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "ConditionalBatchNormalization(num_features=" << num_features_
//                << ", cond_embedding_dim=" << cond_embedding_dim_
//                << ", cond_hidden_dim=" << (fc_cond1_ ? std::to_string(cond_hidden_dim_) : "0 (direct)")
//                << ", eps=" << eps_ << ", momentum=" << momentum_
//                << ", track_running_stats=" << (track_running_stats_ ? "true" : "false") << ")";
//     }
// };
// TORCH_MODULE(ConditionalBatchNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 3;       // Channels in x
//     int64_t cond_embedding_dim = 10; // Dimension of conditioning vector
//     int64_t N = 4;                  // Batch size
//
//     // --- Test Case 1: 4D input x (NCHW), no hidden layer in conditioning net ---
//     std::cout << "--- Test Case 1: 4D input x (NCHW), no hidden layer in cond_net ---" << std::endl;
//     ConditionalBatchNormalization cbn_module1(num_features, cond_embedding_dim, /*cond_hidden_dim=*/0);
//     // std::cout << cbn_module1 << std::endl;
//
//     torch::Tensor x1 = torch::randn({N, num_features, 8, 8});
//     torch::Tensor cond1 = torch::randn({N, cond_embedding_dim});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//     std::cout << "Cond cond1 shape: " << cond1.sizes() << std::endl;
//
//     // Training pass
//     cbn_module1->train();
//     torch::Tensor y1_train = cbn_module1->forward(x1, cond1);
//     std::cout << "Output y1_train shape: " << y1_train.sizes() << std::endl;
//     // Mean/Std of y1_train for a given channel will NOT be 0/1 because gamma/beta are data-dependent
//     // and not necessarily 1/0.
//     std::cout << "y1_train [0,0,:,:] mean: " << y1_train.select(0,0).select(0,0).mean().item<double>() << std::endl;
//     std::cout << "y1_train [0,0,:,:] std:  " << y1_train.select(0,0).select(0,0).std(false).item<double>() << std::endl;
//     std::cout << "Updated running_mean (cbn1): " << cbn_module1->running_mean_ << std::endl;
//
//     // Evaluation pass
//     cbn_module1->eval();
//     torch::Tensor y1_eval = cbn_module1->forward(x1, cond1); // Should use running_mean/var
//     std::cout << "Output y1_eval shape: " << y1_eval.sizes() << std::endl;
//     std::cout << "y1_eval [0,0,:,:] mean: " << y1_eval.select(0,0).select(0,0).mean().item<double>() << std::endl;
//     TORCH_CHECK(!torch::allclose(y1_train.select(0,0).select(0,0).mean(), y1_eval.select(0,0).select(0,0).mean()),
//                 "Train and Eval output means should differ for CBN with track_running_stats.");
//
//
//     // --- Test Case 2: 2D input x (NC), with hidden layer in conditioning net ---
//     std::cout << "\n--- Test Case 2: 2D input x (NC), with hidden layer in cond_net ---" << std::endl;
//     int64_t cond_hidden = 32;
//     ConditionalBatchNormalization cbn_module2(num_features, cond_embedding_dim, cond_hidden);
//     // std::cout << cbn_module2 << std::endl;
//
//     torch::Tensor x2 = torch::randn({N, num_features});
//     torch::Tensor cond2 = torch::randn({N, cond_embedding_dim});
//     std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
//     std::cout << "Cond cond2 shape: " << cond2.sizes() << std::endl;
//
//     cbn_module2->train();
//     torch::Tensor y2_train = cbn_module2->forward(x2, cond2);
//     std::cout << "Output y2_train shape: " << y2_train.sizes() << std::endl;
//
//
//     // --- Test Case 3: Check backward pass (requires gradients) ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     ConditionalBatchNormalization cbn_module3(num_features, cond_embedding_dim, cond_hidden);
//     cbn_module3->train();
//
//     torch::Tensor x3 = torch::randn({N, num_features, 6, 6}, torch::requires_grad());
//     torch::Tensor cond3 = torch::randn({N, cond_embedding_dim}, torch::requires_grad());
//
//     torch::Tensor y3 = cbn_module3->forward(x3, cond3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_cond3 = cond3.grad().defined() && cond3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_fc_out_weight = cbn_module3->fc_cond_out_->weight.grad().defined() &&
//                                      cbn_module3->fc_cond_out_->weight.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for cond3: " << (grad_exists_cond3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for fc_cond_out.weight: " << (grad_exists_fc_out_weight ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_cond3, "No gradient for cond3!");
//     TORCH_CHECK(grad_exists_fc_out_weight, "No gradient for fc_cond_out.weight!");
//
//
//     // Test track_running_stats = false
//     std::cout << "\n--- Test Case 4: No track_running_stats ---" << std::endl;
//     ConditionalBatchNormalization cbn_module4(num_features, cond_embedding_dim, 0, 1e-5, 0.1, false);
//     cbn_module4->train();
//     torch::Tensor y4_train = cbn_module4->forward(x1, cond1);
//     cbn_module4->eval();
//     torch::Tensor y4_eval = cbn_module4->forward(x1, cond1); // Should use batch stats as track_running_stats=false
//     TORCH_CHECK(cbn_module4->running_mean_.defined() == false, "Running mean should not be defined.");
//     TORCH_CHECK(torch::allclose(y4_train, y4_eval),
//                 "Outputs should be same if not tracking running stats, regardless of train/eval mode for CBN.");
//
//
//     std::cout << "\nConditionalBatchNormalization tests finished." << std::endl;
//     return 0;
// }




namespace xt::norm
{
    auto ConditionalBatchNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
