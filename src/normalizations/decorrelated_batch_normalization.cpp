#include "include/normalizations/decorrelated_batch_normalization.h"


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct DecorrelatedBatchNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct DecorrelatedBatchNormalization : torch::nn::ModuleHolder<DecorrelatedBatchNormalizationImpl> {
//     using torch::nn::ModuleHolder<DecorrelatedBatchNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct
// struct DecorrelatedBatchNormalizationImpl : torch::nn::Module {
//     int64_t num_features_; // Number of input features (e.g., C if input is N,C or C*H*W if flattened)
//     double eps_bn_;        // Epsilon for the initial Batch Normalization part
//     double momentum_bn_;   // Momentum for the initial Batch Normalization part
//     bool affine_bn_;       // Whether the initial BN has affine parameters (gamma, beta)
//                            // For DBN, often the affine params of initial BN are disabled or fixed
//                            // as the decorrelation matrix and final affine params handle scaling/shifting.
//     bool affine_final_;    // Whether to apply final affine parameters after decorrelation
//
//     // Standard Batch Normalization components
//     torch::Tensor running_mean_bn_;
//     torch::Tensor running_var_bn_;
//     torch::Tensor gamma_bn_; // Optional affine for BN part
//     torch::Tensor beta_bn_;  // Optional affine for BN part
//     torch::Tensor num_batches_tracked_bn_;
//
//     // Decorrelation components
//     // We learn a linear transformation W for decorrelation.
//     // W should be (num_features_, num_features_)
//     torch::Tensor decorrelation_matrix_W_; // W
//
//     // Final affine parameters (optional, applied after decorrelation)
//     torch::Tensor gamma_final_;
//     torch::Tensor beta_final_;
//
//     // IterNorm specific (for more advanced decorrelation, not fully implemented here for simplicity)
//     // int num_iter_newton_; // Number of Newton's iterations for Sigma^{-1/2}
//
//     DecorrelatedBatchNormalizationImpl(int64_t num_features,
//                                        double eps_bn = 1e-5,
//                                        double momentum_bn = 0.1,
//                                        bool affine_bn = false, // Often false or fixed for DBN's BN part
//                                        bool affine_final = true)
//         : num_features_(num_features),
//           eps_bn_(eps_bn),
//           momentum_bn_(momentum_bn),
//           affine_bn_(affine_bn),
//           affine_final_(affine_final) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//
//         // --- Standard Batch Normalization Part ---
//         // Always track running stats for BN part, as it's standard.
//         running_mean_bn_ = register_buffer("running_mean_bn", torch::zeros({num_features_}));
//         running_var_bn_ = register_buffer("running_var_bn", torch::ones({num_features_}));
//         num_batches_tracked_bn_ = register_buffer("num_batches_tracked_bn", torch::tensor(0, torch::kLong));
//
//         if (affine_bn_) {
//             gamma_bn_ = register_parameter("gamma_bn", torch::ones({num_features_}));
//             beta_bn_ = register_parameter("beta_bn", torch::zeros({num_features_}));
//         }
//
//         // --- Decorrelation Part ---
//         // Initialize W. Ideally, it starts as an identity matrix or close to it.
//         // Or it can be learned from scratch.
//         // For a whitening matrix W (like Sigma^{-1/2}), W W^T = Sigma^{-1}.
//         // If W is initialized as Identity, it means no decorrelation initially.
//         decorrelation_matrix_W_ = register_parameter("decorrelation_matrix_W", torch::eye(num_features_));
//
//         // --- Final Affine Part ---
//         if (affine_final_) {
//             gamma_final_ = register_parameter("gamma_final", torch::ones({num_features_}));
//             beta_final_ = register_parameter("beta_final", torch::zeros({num_features_}));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x is expected to be 2D: (N, C) where C is num_features_.
//         // If x is 4D (N,C,H,W), it should be flattened to (N, C*H*W) before passing here,
//         // or this module needs to handle the reshape.
//         // For this implementation, let's assume x is already (N, num_features_).
//
//         TORCH_CHECK(x.dim() == 2, "Input tensor x must be 2D (Batch, Features). Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Number of input features (dim 1) mismatch. Expected ", num_features_,
//                     ", but got ", x.size(1));
//
//         // --- 1. Standard Batch Normalization ---
//         torch::Tensor current_mean_bn;
//         torch::Tensor current_var_bn;
//
//         if (this->is_training()) {
//             // Mean and Var are calculated per feature across the batch
//             current_mean_bn = x.mean(/*dim=*/0, /*keepdim=false*/ false); // Shape (num_features_)
//             // current_var_bn = x.var(/*dim=*/0, /*unbiased=*/false, /*keepdim=false*/ false); // Shape (num_features_)
//             // More stable variance: E[X^2] - (E[X])^2
//             current_var_bn = (x - current_mean_bn).pow(2).mean(/*dim=*/0, /*keepdim=false*/false);
//
//
//             running_mean_bn_ = (1.0 - momentum_bn_) * running_mean_bn_ + momentum_bn_ * current_mean_bn.detach(); // Detach for running stats
//             running_var_bn_  = (1.0 - momentum_bn_) * running_var_bn_  + momentum_bn_ * current_var_bn.detach();
//             if (num_batches_tracked_bn_) num_batches_tracked_bn_ += 1;
//         } else {
//             current_mean_bn = running_mean_bn_;
//             current_var_bn = running_var_bn_;
//         }
//
//         torch::Tensor x_bn = (x - current_mean_bn) / torch::sqrt(current_var_bn + eps_bn_);
//
//         if (affine_bn_) {
//             x_bn = x_bn * gamma_bn_ + beta_bn_;
//         }
//
//         // --- 2. Decorrelation Step ---
//         // x_bn is (N, num_features_). decorrelation_matrix_W_ is (num_features_, num_features_).
//         // We want to transform each feature vector: y = x_bn @ W^T  (or W @ x_bn^T then transpose)
//         // If W is Sigma^{-1/2}, then y = Sigma^{-1/2} * x_bn
//         // The DBN paper formulation is often y = W(x - mu)/sigma.
//         // If x_bn already incorporates (x-mu)/sigma, then y = W * x_bn
//         // Let's assume y_i = sum_j W_ij * x_bn_j. This is x_bn @ W.T
//         torch::Tensor x_decorrelated = torch::matmul(x_bn, decorrelation_matrix_W_.t()); // (N, num_features_)
//
//         // Note on IterNorm:
//         // A full IterNorm would compute covariance of x_bn: Sigma = (x_bn.T @ x_bn) / N
//         // Then find Sigma^{-1/2} using Newton's iteration:
//         // Y_0 = I
//         // Y_{k+1} = 0.5 * (Y_k + (Y_k @ Sigma)^{-T})  (or similar variants for Sigma^{-1/2})
//         // W = Y_final (or Sigma @ Y_final for Sigma^{1/2} if W is applied as W*x)
//         // This is computationally intensive for each forward pass if not carefully optimized or if N is large.
//         // Learning W directly is a simplification. The gradient will try to make W a decorrelating transform.
//
//         // --- 3. Optional Final Affine Transformation ---
//         if (affine_final_) {
//             // gamma_final_ and beta_final_ are (num_features_). They broadcast.
//             x_decorrelated = x_decorrelated * gamma_final_ + beta_final_;
//         }
//
//         return x_decorrelated;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "DecorrelatedBatchNormalization(num_features=" << num_features_
//                << ", eps_bn=" << eps_bn_ << ", momentum_bn=" << momentum_bn_
//                << ", affine_bn=" << (affine_bn_ ? "true" : "false")
//                << ", affine_final=" << (affine_final_ ? "true" : "false") << ")";
//     }
// };
// TORCH_MODULE(DecorrelatedBatchNormalization);
//
//
// // Helper function to compute covariance matrix for checking
// torch::Tensor cov(const torch::Tensor& x) { // x is (Batch, Features)
//     auto x_mean_sub = x - x.mean(0, true);
//     auto N = x.size(0);
//     return torch::matmul(x_mean_sub.t(), x_mean_sub) / (N - 1); // Sample covariance
// }
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 5; // Use a small number of features for easier covariance inspection
//     int64_t N = 100;         // Batch size
//
//     DecorrelatedBatchNormalization dbn_module(num_features,
//                                              /*eps_bn=*/1e-5,
//                                              /*momentum_bn=*/0.1,
//                                              /*affine_bn=*/false, // DBN often sets this false
//                                              /*affine_final=*/true);
//     // std::cout << dbn_module << std::endl;
//
//     // Create correlated input data for x
//     torch::Tensor base = torch::randn({N, num_features});
//     torch::Tensor correl_matrix = torch::tensor({{1.0, 0.5, 0.2, 0.0, 0.1},
//                                                  {0.5, 1.0, 0.6, 0.1, 0.0},
//                                                  {0.2, 0.6, 1.0, 0.4, 0.3},
//                                                  {0.0, 0.1, 0.4, 1.0, 0.7},
//                                                  {0.1, 0.0, 0.3, 0.7, 1.0}}, torch::kFloat);
//     // Cholesky decomposition to create correlated data: data = base @ L.T where L L.T = correl_matrix
//     torch::Tensor L = torch::linalg::cholesky(correl_matrix);
//     torch::Tensor x = torch::matmul(base, L.t()) * 2.0 + 3.0; // Scale and shift
//
//     std::cout << "Input x shape: " << x.sizes() << std::endl;
//     std::cout << "Covariance of input x (approx): \n" << cov(x) << std::endl;
//
//
//     // --- Training Loop (simplified) ---
//     // In a real scenario, you'd train W. Here we just do a few forward passes.
//     // The learned W aims to make Cov(W * x_bn) diagonal.
//     dbn_module->train();
//     torch::optim::SGD optimizer(dbn_module->parameters(), /*lr=*/0.1); // Dummy optimizer
//
//     std::cout << "\nInitial W:\n" << dbn_module->decorrelation_matrix_W_ << std::endl;
//
//     torch::Tensor output_before_train_eval_mode;
//     {
//         torch::NoGradGuard no_grad; // Get output in eval mode before any training
//         dbn_module->eval();
//         output_before_train_eval_mode = dbn_module->forward(x);
//     }
//
//
//     std::cout << "\n--- Simulating Training (few steps) ---" << std::endl;
//     for (int i = 0; i < 20; ++i) { // More iterations might show W changing
//         optimizer.zero_grad();
//         dbn_module->train(); // Set to train mode
//         torch::Tensor output_train = dbn_module->forward(x);
//
//         // A loss that encourages decorrelation could be:
//         // Sum of off-diagonal elements of Cov(output_train), or log-likelihood under Gaussian.
//         // For simplicity, just use mean as a dummy loss to get gradients.
//         torch::Tensor loss = output_train.mean(); // Dummy loss
//         if (i % 5 == 0) {
//            std::cout << "Step " << i << ", Dummy Loss: " << loss.item<float>();
//            torch::Tensor x_bn_intermediate = (x - dbn_module->running_mean_bn_) / torch::sqrt(dbn_module->running_var_bn_ + dbn_module->eps_bn_);
//            if (dbn_module->affine_bn_) x_bn_intermediate = x_bn_intermediate * dbn_module->gamma_bn_ + dbn_module->beta_bn_;
//            torch::Tensor x_decor_intermediate = torch::matmul(x_bn_intermediate, dbn_module->decorrelation_matrix_W_.t());
//            std::cout << ", Cov(x_decorrelated_before_final_affine) off-diag sum: " << (cov(x_decor_intermediate).sum() - cov(x_decor_intermediate).diag().sum()).abs().item<float>() << std::endl;
//
//         }
//         loss.backward();
//         optimizer.step();
//     }
//     std::cout << "Trained W (should have changed from identity):\n" << dbn_module->decorrelation_matrix_W_ << std::endl;
//
//
//     // --- Evaluation pass ---
//     dbn_module->eval();
//     torch::Tensor output_eval = dbn_module->forward(x);
//     std::cout << "\nOutput_eval shape: " << output_eval.sizes() << std::endl;
//
//     std::cout << "Covariance of output_eval (should be more diagonal if W learned well, and scaled by gamma_final^2): \n"
//               << cov(output_eval) << std::endl;
//
//     // Check mean and variance of output_eval (should be approx beta_final and gamma_final^2 if W is orthogonal)
//     std::cout << "Output_eval mean (per feature, approx beta_final): \n" << output_eval.mean(0) << std::endl;
//     std::cout << "Output_eval var (per feature, approx gamma_final^2): \n" << output_eval.var(0, false) << std::endl;
//
//     TORCH_CHECK(!torch::allclose(output_before_train_eval_mode, output_eval, 1e-3, 1e-3),
//                 "Output before and after dummy training should differ due to W update.");
//
//
//     std::cout << "\nDecorrelatedBatchNormalization tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    DecorrelatedBatchNorm::DecorrelatedBatchNorm(int64_t num_features,
                                                 double eps_bn,
                                                 double momentum_bn,
                                                 bool affine_bn, // Often false or fixed for DBN's BN part
                                                 bool affine_final)
        : num_features_(num_features),
          eps_bn_(eps_bn),
          momentum_bn_(momentum_bn),
          affine_bn_(affine_bn),
          affine_final_(affine_final)
    {
        TORCH_CHECK(num_features > 0, "num_features must be positive.");

        // --- Standard Batch Normalization Part ---
        // Always track running stats for BN part, as it's standard.
        running_mean_bn_ = register_buffer("running_mean_bn", torch::zeros({num_features_}));
        running_var_bn_ = register_buffer("running_var_bn", torch::ones({num_features_}));
        num_batches_tracked_bn_ = register_buffer("num_batches_tracked_bn", torch::tensor(0, torch::kLong));

        if (affine_bn_)
        {
            gamma_bn_ = register_parameter("gamma_bn", torch::ones({num_features_}));
            beta_bn_ = register_parameter("beta_bn", torch::zeros({num_features_}));
        }

        // --- Decorrelation Part ---
        // Initialize W. Ideally, it starts as an identity matrix or close to it.
        // Or it can be learned from scratch.
        // For a whitening matrix W (like Sigma^{-1/2}), W W^T = Sigma^{-1}.
        // If W is initialized as Identity, it means no decorrelation initially.
        decorrelation_matrix_W_ = register_parameter("decorrelation_matrix_W", torch::eye(num_features_));

        // --- Final Affine Part ---
        if (affine_final_)
        {
            gamma_final_ = register_parameter("gamma_final", torch::ones({num_features_}));
            beta_final_ = register_parameter("beta_final", torch::zeros({num_features_}));
        }
    }

    auto DecorrelatedBatchNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);


        // Input x is expected to be 2D: (N, C) where C is num_features_.
        // If x is 4D (N,C,H,W), it should be flattened to (N, C*H*W) before passing here,
        // or this module needs to handle the reshape.
        // For this implementation, let's assume x is already (N, num_features_).

        TORCH_CHECK(x.dim() == 2, "Input tensor x must be 2D (Batch, Features). Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Number of input features (dim 1) mismatch. Expected ", num_features_,
                    ", but got ", x.size(1));

        // --- 1. Standard Batch Normalization ---
        torch::Tensor current_mean_bn;
        torch::Tensor current_var_bn;

        if (this->is_training())
        {
            // Mean and Var are calculated per feature across the batch
            current_mean_bn = x.mean(/*dim=*/0, /*keepdim=false*/ false); // Shape (num_features_)
            // current_var_bn = x.var(/*dim=*/0, /*unbiased=*/false, /*keepdim=false*/ false); // Shape (num_features_)
            // More stable variance: E[X^2] - (E[X])^2
            current_var_bn = (x - current_mean_bn).pow(2).mean(/*dim=*/0, /*keepdim=false*/false);


            running_mean_bn_ = (1.0 - momentum_bn_) * running_mean_bn_ + momentum_bn_ * current_mean_bn.detach();
            // Detach for running stats
            running_var_bn_ = (1.0 - momentum_bn_) * running_var_bn_ + momentum_bn_ * current_var_bn.detach();

            //TODO SOME BUGS MIGHT RAISE HERE
            // original code was :
            // if (num_batches_tracked_bn_)
            // {
            //     // Check if defined
            //     num_batches_tracked_bn_ += 1;
            // }

            if (num_batches_tracked_bn_[0].item<int64>() == 0)
            {
                // Check if defined
                num_batches_tracked_bn_ += 1;
            }
        }
        else
        {
            current_mean_bn = running_mean_bn_;
            current_var_bn = running_var_bn_;
        }

        torch::Tensor x_bn = (x - current_mean_bn) / torch::sqrt(current_var_bn + eps_bn_);

        if (affine_bn_)
        {
            x_bn = x_bn * gamma_bn_ + beta_bn_;
        }

        // --- 2. Decorrelation Step ---
        // x_bn is (N, num_features_). decorrelation_matrix_W_ is (num_features_, num_features_).
        // We want to transform each feature vector: y = x_bn @ W^T  (or W @ x_bn^T then transpose)
        // If W is Sigma^{-1/2}, then y = Sigma^{-1/2} * x_bn
        // The DBN paper formulation is often y = W(x - mu)/sigma.
        // If x_bn already incorporates (x-mu)/sigma, then y = W * x_bn
        // Let's assume y_i = sum_j W_ij * x_bn_j. This is x_bn @ W.T
        torch::Tensor x_decorrelated = torch::matmul(x_bn, decorrelation_matrix_W_.t()); // (N, num_features_)

        // Note on IterNorm:
        // A full IterNorm would compute covariance of x_bn: Sigma = (x_bn.T @ x_bn) / N
        // Then find Sigma^{-1/2} using Newton's iteration:
        // Y_0 = I
        // Y_{k+1} = 0.5 * (Y_k + (Y_k @ Sigma)^{-T})  (or similar variants for Sigma^{-1/2})
        // W = Y_final (or Sigma @ Y_final for Sigma^{1/2} if W is applied as W*x)
        // This is computationally intensive for each forward pass if not carefully optimized or if N is large.
        // Learning W directly is a simplification. The gradient will try to make W a decorrelating transform.

        // --- 3. Optional Final Affine Transformation ---
        if (affine_final_)
        {
            // gamma_final_ and beta_final_ are (num_features_). They broadcast.
            x_decorrelated = x_decorrelated * gamma_final_ + beta_final_;
        }

        return x_decorrelated;
    }
}
