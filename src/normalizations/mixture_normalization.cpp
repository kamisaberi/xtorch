#include "include/normalizations/mixture_normalization.h"

//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // Forward declaration for the Impl struct
// struct MixtureNormalizationImpl;
//
// // The main module struct that users will interact with.
// struct MixtureNormalization : torch::nn::ModuleHolder<MixtureNormalizationImpl> {
//     using torch::nn::ModuleHolder<MixtureNormalizationImpl>::ModuleHolder;
//
//     torch::Tensor forward(torch::Tensor x) {
//         return impl_->forward(x);
//     }
// };
//
// // The implementation struct for MixtureNormalization
// struct MixtureNormalizationImpl : torch::nn::Module {
//     int64_t num_features_;
//     double eps_bn_;
//     double momentum_bn_;
//     double eps_in_;
//     bool affine_bn_; // Whether the BN component has its own affine parameters
//     bool affine_in_; // Whether the IN component has its own affine parameters
//     // Note: The final output will be a mix. If both are true, it's a mix of affined outputs.
//     // One could also have a single final affine transformation after mixing. For simplicity,
//     // let's assume the sub-normalizers handle their own affine if enabled.
//
//     // Batch Normalization components
//     torch::Tensor running_mean_bn_;
//     torch::Tensor running_var_bn_;
//     torch::Tensor gamma_bn_;
//     torch::Tensor beta_bn_;
//     torch::Tensor num_batches_tracked_bn_;
//
//     // Instance Normalization components (no running stats, only optional affine)
//     torch::Tensor gamma_in_;
//     torch::Tensor beta_in_;
//
//     // Learnable mixing parameters (lambda)
//     // We'll learn `lambda_raw_` and pass it through sigmoid to keep it in [0, 1]
//     torch::Tensor lambda_raw_; // Per-channel mixing weights
//
//     MixtureNormalizationImpl(int64_t num_features,
//                              double eps_bn = 1e-5, double momentum_bn = 0.1, bool affine_bn = true,
//                              double eps_in = 1e-5, bool affine_in = true)
//         : num_features_(num_features),
//           eps_bn_(eps_bn), momentum_bn_(momentum_bn), affine_bn_(affine_bn),
//           eps_in_(eps_in), affine_in_(affine_in) {
//
//         TORCH_CHECK(num_features > 0, "num_features must be positive.");
//
//         // --- Batch Normalization Part ---
//         running_mean_bn_ = register_buffer("running_mean_bn", torch::zeros({num_features_}));
//         running_var_bn_ = register_buffer("running_var_bn", torch::ones({num_features_}));
//         num_batches_tracked_bn_ = register_buffer("num_batches_tracked_bn", torch::tensor(0, torch::kLong));
//
//         if (affine_bn_) {
//             gamma_bn_ = register_parameter("gamma_bn", torch::ones({num_features_}));
//             beta_bn_ = register_parameter("beta_bn", torch::zeros({num_features_}));
//         }
//
//         // --- Instance Normalization Part ---
//         if (affine_in_) {
//             gamma_in_ = register_parameter("gamma_in", torch::ones({num_features_}));
//             beta_in_ = register_parameter("beta_in", torch::zeros({num_features_}));
//         }
//
//         // --- Mixing Parameter ---
//         // Initialize lambda_raw_ s.t. initial lambda is 0.5 (sigmoid(0) = 0.5)
//         // Or initialize to favor one, e.g. sigmoid(large_negative) -> 0 (favors IN)
//         // or sigmoid(large_positive) -> 1 (favors BN). Let's start with 0.5.
//         lambda_raw_ = register_parameter("lambda_raw", torch::zeros({1, num_features_, 1, 1})); // For NCHW broadcasting
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Input x expected to be (N, C, D1, D2, ...) e.g., 4D (N,C,H,W)
//         TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got shape ", x.sizes());
//         TORCH_CHECK(x.size(1) == num_features_,
//                     "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));
//
//         int64_t N = x.size(0);
//         // --- Output of Batch Normalization (y_bn) ---
//         torch::Tensor y_bn;
//         {
//             torch::Tensor current_mean_bn, current_var_bn;
//             std::vector<int64_t> reduce_dims_bn;
//             reduce_dims_bn.push_back(0);
//             for (int64_t i = 2; i < x.dim(); ++i) reduce_dims_bn.push_back(i);
//
//             if (this->is_training()) {
//                 current_mean_bn = x.mean(reduce_dims_bn, false);
//                 std::vector<int64_t> view_shape_mean(x.dim(), 1); view_shape_mean[1] = num_features_;
//                 current_var_bn = (x - current_mean_bn.view(view_shape_mean)).pow(2).mean(reduce_dims_bn, false);
//
//                 running_mean_bn_ = (1.0 - momentum_bn_) * running_mean_bn_ + momentum_bn_ * current_mean_bn.detach();
//                 running_var_bn_  = (1.0 - momentum_bn_) * running_var_bn_  + momentum_bn_ * current_var_bn.detach();
//                 if (num_batches_tracked_bn_) num_batches_tracked_bn_ += 1;
//             } else {
//                 current_mean_bn = running_mean_bn_;
//                 current_var_bn = running_var_bn_;
//             }
//             std::vector<int64_t> bn_param_view_shape(x.dim(), 1); bn_param_view_shape[1] = num_features_;
//             y_bn = (x - current_mean_bn.view(bn_param_view_shape)) /
//                    torch::sqrt(current_var_bn.view(bn_param_view_shape) + eps_bn_);
//             if (affine_bn_) {
//                 y_bn = y_bn * gamma_bn_.view(bn_param_view_shape) + beta_bn_.view(bn_param_view_shape);
//             }
//         }
//
//         // --- Output of Instance Normalization (y_in) ---
//         torch::Tensor y_in;
//         {
//             if (x.dim() > 2) {
//                 std::vector<int64_t> reduce_dims_in;
//                 for (int64_t i = 2; i < x.dim(); ++i) reduce_dims_in.push_back(i);
//                 auto mean_in = x.mean(reduce_dims_in, true);
//                 auto var_in = x.var(reduce_dims_in, false, true);
//                 y_in = (x - mean_in) / torch::sqrt(var_in + eps_in_);
//             } else { // x.dim() == 2 (N,C)
//                 y_in = torch::zeros_like(x); // IN of a single point is 0
//             }
//             if (affine_in_) {
//                 std::vector<int64_t> in_param_view_shape(x.dim(), 1); in_param_view_shape[1] = num_features_;
//                 y_in = y_in * gamma_in_.view(in_param_view_shape) + beta_in_.view(in_param_view_shape);
//             }
//         }
//
//         // --- Mix the outputs ---
//         torch::Tensor lambda_ = torch::sigmoid(lambda_raw_); // Shape (1, C, 1, 1)
//
//         // lambda_ will broadcast with y_bn and y_in if they are (N,C,H,W)
//         torch::Tensor output = lambda_ * y_bn + (1.0 - lambda_) * y_in;
//
//         return output;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "MixtureNormalization(num_features=" << num_features_
//                << ", eps_bn=" << eps_bn_ << ", momentum_bn=" << momentum_bn_ << ", affine_bn=" << affine_bn_
//                << ", eps_in=" << eps_in_ << ", affine_in=" << affine_in_ << ")";
//     }
// };
// TORCH_MODULE(MixtureNormalization);
//
//
// // --- Example Usage ---
// int main() {
//     torch::manual_seed(0);
//
//     int64_t num_features = 32;
//     int64_t N = 4, H = 16, W = 16;
//
//     // --- Test Case 1: MixtureNormalization with defaults ---
//     std::cout << "--- Test Case 1: MixtureNormalization defaults ---" << std::endl;
//     MixtureNormalization mixnorm_module1(num_features);
//     // std::cout << mixnorm_module1 << std::endl;
//     std::cout << "Initial lambda_raw (all zeros): " << mixnorm_module1->lambda_raw_.mean().item<float>() << std::endl;
//     std::cout << "Initial lambda (sigmoid(0)=0.5): " << torch::sigmoid(mixnorm_module1->lambda_raw_).mean().item<float>() << std::endl;
//
//
//     torch::Tensor x1 = torch::randn({N, num_features, H, W});
//     std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
//
//     // Training pass
//     mixnorm_module1->train();
//     torch::Tensor y1_train = mixnorm_module1->forward(x1);
//     std::cout << "Output y1_train shape: " << y1_train.sizes() << std::endl;
//     std::cout << "y1_train mean (all): " << y1_train.mean().item<double>() << std::endl;
//
//     // Evaluation pass
//     mixnorm_module1->eval();
//     torch::Tensor y1_eval = mixnorm_module1->forward(x1);
//     std::cout << "Output y1_eval shape: " << y1_eval.sizes() << std::endl;
//     std::cout << "y1_eval mean (all): " << y1_eval.mean().item<double>() << std::endl;
//     TORCH_CHECK(!torch::allclose(y1_train.mean(), y1_eval.mean()),
//                 "Train and Eval output means should differ due to BN part.");
//
//
//     // --- Test Case 2: Forcing lambda towards BN or IN ---
//     std::cout << "\n--- Test Case 2: Forcing lambda ---" << std::endl;
//     MixtureNormalization mixnorm_module2(num_features, 1e-5, 0.1, false, 1e-5, false); // No sub-affine for clarity
//     torch::Tensor x2 = torch::randn({N, num_features, H, W}) * 5 + 10; // Input with distinct stats
//
//     // Force lambda to favor BN (lambda -> 1)
//     mixnorm_module2->lambda_raw_.data().fill_(10.0); // sigmoid(10) is close to 1
//     std::cout << "Lambda (BN favored): " << torch::sigmoid(mixnorm_module2->lambda_raw_)[0][0][0][0].item<float>() << std::endl;
//     mixnorm_module2->eval(); // Use running stats for BN part
//     torch::Tensor y2_bn_favored = mixnorm_module2->forward(x2);
//
//     // Force lambda to favor IN (lambda -> 0)
//     mixnorm_module2->lambda_raw_.data().fill_(-10.0); // sigmoid(-10) is close to 0
//     std::cout << "Lambda (IN favored): " << torch::sigmoid(mixnorm_module2->lambda_raw_)[0][0][0][0].item<float>() << std::endl;
//     mixnorm_module2->eval(); // Mode doesn't change IN part's core stats
//     torch::Tensor y2_in_favored = mixnorm_module2->forward(x2);
//
//     // The outputs should be different
//     std::cout << "Mean of BN-favored output: " << y2_bn_favored.mean().item<double>() << std::endl;
//     std::cout << "Mean of IN-favored output: " << y2_in_favored.mean().item<double>() << std::endl;
//     TORCH_CHECK(!torch::allclose(y2_bn_favored.mean(), y2_in_favored.mean()),
//                 "BN-favored and IN-favored outputs should differ.");
//
//
//     // --- Test Case 3: Check backward pass ---
//     std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
//     MixtureNormalization mixnorm_module3(num_features);
//     mixnorm_module3->train();
//
//     torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
//     torch::Tensor y3 = mixnorm_module3->forward(x3);
//     torch::Tensor loss = y3.mean();
//     loss.backward();
//
//     bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_lambda_raw = mixnorm_module3->lambda_raw_.grad().defined() &&
//                                   mixnorm_module3->lambda_raw_.grad().abs().sum().item<double>() > 0;
//     bool grad_exists_gamma_bn = mixnorm_module3->gamma_bn_.grad().defined() && // if affine_bn=true
//                                 mixnorm_module3->gamma_bn_.grad().abs().sum().item<double>() > 0;
//
//     std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for lambda_raw: " << (grad_exists_lambda_raw ? "true" : "false") << std::endl;
//     std::cout << "Gradient exists for gamma_bn: " << (grad_exists_gamma_bn ? "true" : "false") << std::endl;
//
//     TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
//     TORCH_CHECK(grad_exists_lambda_raw, "No gradient for lambda_raw!");
//     TORCH_CHECK(grad_exists_gamma_bn, "No gradient for gamma_bn!");
//
//
//     std::cout << "\nMixtureNormalization tests finished." << std::endl;
//     return 0;
// }


namespace xt::norm
{
    MixtureNorm::MixtureNorm(int64_t num_features,
                             double eps_bn, double momentum_bn, bool affine_bn,
                             double eps_in, bool affine_in)
        : num_features_(num_features),
          eps_bn_(eps_bn), momentum_bn_(momentum_bn), affine_bn_(affine_bn),
          eps_in_(eps_in), affine_in_(affine_in)
    {
        TORCH_CHECK(num_features > 0, "num_features must be positive.");

        // --- Batch Normalization Part ---
        running_mean_bn_ = register_buffer("running_mean_bn", torch::zeros({num_features_}));
        running_var_bn_ = register_buffer("running_var_bn", torch::ones({num_features_}));
        num_batches_tracked_bn_ = register_buffer("num_batches_tracked_bn", torch::tensor(0, torch::kLong));

        if (affine_bn_)
        {
            gamma_bn_ = register_parameter("gamma_bn", torch::ones({num_features_}));
            beta_bn_ = register_parameter("beta_bn", torch::zeros({num_features_}));
        }

        // --- Instance Normalization Part ---
        if (affine_in_)
        {
            gamma_in_ = register_parameter("gamma_in", torch::ones({num_features_}));
            beta_in_ = register_parameter("beta_in", torch::zeros({num_features_}));
        }

        // --- Mixing Parameter ---
        // Initialize lambda_raw_ s.t. initial lambda is 0.5 (sigmoid(0) = 0.5)
        // Or initialize to favor one, e.g. sigmoid(large_negative) -> 0 (favors IN)
        // or sigmoid(large_positive) -> 1 (favors BN). Let's start with 0.5.
        lambda_raw_ = register_parameter("lambda_raw", torch::zeros({1, num_features_, 1, 1})); // For NCHW broadcasting
    }

    auto MixtureNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto x = std::any_cast<torch::Tensor>(tensors_[0]);

        // Input x expected to be (N, C, D1, D2, ...) e.g., 4D (N,C,H,W)
        TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));

        int64_t N = x.size(0);
        // --- Output of Batch Normalization (y_bn) ---
        torch::Tensor y_bn;
        {
            torch::Tensor current_mean_bn, current_var_bn;
            std::vector<int64_t> reduce_dims_bn;
            reduce_dims_bn.push_back(0);
            for (int64_t i = 2; i < x.dim(); ++i) reduce_dims_bn.push_back(i);

            if (this->is_training())
            {
                current_mean_bn = x.mean(reduce_dims_bn, false);
                std::vector<int64_t> view_shape_mean(x.dim(), 1);
                view_shape_mean[1] = num_features_;
                current_var_bn = (x - current_mean_bn.view(view_shape_mean)).pow(2).mean(reduce_dims_bn, false);

                running_mean_bn_ = (1.0 - momentum_bn_) * running_mean_bn_ + momentum_bn_ * current_mean_bn.detach();
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
            std::vector<int64_t> bn_param_view_shape(x.dim(), 1);
            bn_param_view_shape[1] = num_features_;
            y_bn = (x - current_mean_bn.view(bn_param_view_shape)) /
                torch::sqrt(current_var_bn.view(bn_param_view_shape) + eps_bn_);
            if (affine_bn_)
            {
                y_bn = y_bn * gamma_bn_.view(bn_param_view_shape) + beta_bn_.view(bn_param_view_shape);
            }
        }

        // --- Output of Instance Normalization (y_in) ---
        torch::Tensor y_in;
        {
            if (x.dim() > 2)
            {
                std::vector<int64_t> reduce_dims_in;
                for (int64_t i = 2; i < x.dim(); ++i) reduce_dims_in.push_back(i);
                auto mean_in = x.mean(reduce_dims_in, true);
                auto var_in = x.var(reduce_dims_in, false, true);
                y_in = (x - mean_in) / torch::sqrt(var_in + eps_in_);
            }
            else
            {
                // x.dim() == 2 (N,C)
                y_in = torch::zeros_like(x); // IN of a single point is 0
            }
            if (affine_in_)
            {
                std::vector<int64_t> in_param_view_shape(x.dim(), 1);
                in_param_view_shape[1] = num_features_;
                y_in = y_in * gamma_in_.view(in_param_view_shape) + beta_in_.view(in_param_view_shape);
            }
        }

        // --- Mix the outputs ---
        torch::Tensor lambda_ = torch::sigmoid(lambda_raw_); // Shape (1, C, 1, 1)

        // lambda_ will broadcast with y_bn and y_in if they are (N,C,H,W)
        torch::Tensor output = lambda_ * y_bn + (1.0 - lambda_) * y_in;

        return output;
    }
}
