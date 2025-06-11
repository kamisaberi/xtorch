#include "include/normalizations/online_normalization.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct OnlineNormalizationImpl;

// The main module struct that users will interact with.
struct OnlineNormalization : torch::nn::ModuleHolder<OnlineNormalizationImpl> {
    using torch::nn::ModuleHolder<OnlineNormalizationImpl>::ModuleHolder;

    torch::Tensor forward(torch::Tensor x) {
        return impl_->forward(x);
    }
};

// The implementation struct for OnlineNormalization
struct OnlineNormalizationImpl : torch::nn::Module {
    int64_t num_features_;
    double eps_;
    double momentum_mu_;    // Momentum for online mean update
    double momentum_sigma_; // Momentum for online variance update
    bool affine_gn_;        // Whether the g(.) affirmative part has learnable alpha, beta

    // Learnable parameters for the affirmative part g(x) = alpha * x + beta
    torch::Tensor alpha_; // Scale for g(x)
    torch::Tensor beta_;  // Shift for g(x)

    // Buffers for online statistics (for h(y) part)
    // These are E[g(x)] and Var[g(x)]
    torch::Tensor online_mu_;
    torch::Tensor online_sigma_sq_; // online variance

    // For controlling the update during the very first batch
    torch::Tensor num_batches_tracked_; // Similar to BN's, but for initializing online stats

    OnlineNormalizationImpl(int64_t num_features,
                            double eps = 1e-5,
                            double momentum_mu = 0.1,   // Paper suggests 0.1 or 0.01 for mu
                            double momentum_sigma = 0.1, // Paper suggests 0.1 or 0.001 for sigma
                            bool affine_gn = true)
        : num_features_(num_features),
          eps_(eps),
          momentum_mu_(momentum_mu),
          momentum_sigma_(momentum_sigma),
          affine_gn_(affine_gn) {

        TORCH_CHECK(num_features > 0, "num_features must be positive.");

        if (affine_gn_) {
            alpha_ = register_parameter("alpha", torch::ones({1, num_features_, 1, 1})); // For NCHW broadcasting
            beta_  = register_parameter("beta",  torch::zeros({1, num_features_, 1, 1}));
        } else {
            // If not affine, g(x) = x. Register non-learnable identity transform.
            alpha_ = register_buffer("alpha_const", torch::ones({1, num_features_, 1, 1}));
            beta_  = register_buffer("beta_const",  torch::zeros({1, num_features_, 1, 1}));
        }

        // Initialize online statistics buffers
        online_mu_ = register_buffer("online_mu", torch::zeros({1, num_features_, 1, 1}));
        online_sigma_sq_ = register_buffer("online_sigma_sq", torch::ones({1, num_features_, 1, 1})); // Initialize variance to 1

        num_batches_tracked_ = register_buffer("num_batches_tracked", torch::tensor(0, torch::kLong));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input x is expected to be 4D: (N, C, H, W)
        TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));

        // --- 1. Affirmative part: g(x) = alpha * x + beta ---
        // alpha_ and beta_ are (1,C,1,1) and broadcast with x (N,C,H,W)
        torch::Tensor y = alpha_ * x + beta_;

        // --- 2. Online estimation of E[y] and Var[y] and Normalization h(y) ---
        // During training, update online stats and use them.
        // During eval, use the frozen online stats.
        // The key is that online_mu and online_sigma_sq are treated as constants for grad computation of h(y).
        // This is implicitly handled by using .detach() when updating them,
        // or by PyTorch's default behavior for buffers if they are not direct inputs to grad-requiring ops.

        torch::Tensor current_batch_mu;
        torch::Tensor current_batch_sigma_sq;

        // Statistics are computed over N, H, W for each channel C
        std::vector<int64_t> reduce_dims_stats = {0, 2, 3}; // Batch, Height, Width

        if (this->is_training()) {
            // Calculate statistics of y from the current batch
            current_batch_mu = y.mean(reduce_dims_stats, /*keepdim=*/true); // Shape (1, C, 1, 1)
            current_batch_sigma_sq = y.var(reduce_dims_stats, /*unbiased=*/false, /*keepdim=*/true); // Shape (1, C, 1, 1)

            // Update online statistics
            if (num_batches_tracked_.item<int64_t>() == 0) {
                // First batch: initialize online stats directly from batch stats
                online_mu_.copy_(current_batch_mu.detach());
                online_sigma_sq_.copy_(current_batch_sigma_sq.detach());
            } else {
                online_mu_ = (1.0 - momentum_mu_) * online_mu_.detach() + momentum_mu_ * current_batch_mu.detach();
                online_sigma_sq_ = (1.0 - momentum_sigma_) * online_sigma_sq_.detach() + momentum_sigma_ * current_batch_sigma_sq.detach();
            }
            num_batches_tracked_ += 1;
        }
        // For normalization, always use the current (potentially updated if training) online_mu_ and online_sigma_sq_
        // These are detached during update, so they act as constants for the normalization step's gradient.

        torch::Tensor y_normalized = (y - online_mu_) / torch::sqrt(online_sigma_sq_ + eps_);

        return y_normalized;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "OnlineNormalization(num_features=" << num_features_
               << ", eps=" << eps_
               << ", momentum_mu=" << momentum_mu_ << ", momentum_sigma=" << momentum_sigma_
               << ", affine_g(x)=" << (affine_gn_ ? "true" : "false") << ")";
    }
};
TORCH_MODULE(OnlineNormalization);


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    int64_t num_features = 3; // Use fewer features for easier inspection
    int64_t N = 2, H = 4, W = 4; // Small batch for testing

    // --- Test Case 1: OnlineNormalization with defaults ---
    std::cout << "--- Test Case 1: OnlineNormalization defaults ---" << std::endl;
    OnlineNormalization onlinenorm_module1(num_features);
    // std::cout << onlinenorm_module1 << std::endl;

    torch::Tensor x1 = torch::randn({N, num_features, H, W}) * 2.0 + 3.0; // Input with some scale/shift
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;

    std::cout << "Initial online_mu: " << onlinenorm_module1->online_mu_.slice(1,0,1) << std::endl;
    std::cout << "Initial online_sigma_sq: " << onlinenorm_module1->online_sigma_sq_.slice(1,0,1) << std::endl;

    // Training pass 1
    onlinenorm_module1->train();
    torch::Tensor y1_train_pass1 = onlinenorm_module1->forward(x1);
    std::cout << "Output y1_train_pass1 shape: " << y1_train_pass1.sizes() << std::endl;
    std::cout << "y1_train_pass1 mean (channel 0): " << y1_train_pass1.select(1,0).mean().item<double>() << std::endl; // Should be ~0
    std::cout << "y1_train_pass1 std (channel 0): " << y1_train_pass1.select(1,0).std(false).item<double>() << std::endl; // Should be ~1

    std::cout << "Online_mu after pass 1: " << onlinenorm_module1->online_mu_.slice(1,0,1) << std::endl;
    std::cout << "Online_sigma_sq after pass 1: " << onlinenorm_module1->online_sigma_sq_.slice(1,0,1) << std::endl;
    std::cout << "Num batches tracked: " << onlinenorm_module1->num_batches_tracked_.item<int64_t>() << std::endl;


    // Training pass 2 with different input
    torch::Tensor x1_pass2 = torch::randn({N, num_features, H, W}) * 0.5 - 1.0;
    torch::Tensor y1_train_pass2 = onlinenorm_module1->forward(x1_pass2);
    std::cout << "\nOnline_mu after pass 2: " << onlinenorm_module1->online_mu_.slice(1,0,1) << std::endl;
    std::cout << "Online_sigma_sq after pass 2: " << onlinenorm_module1->online_sigma_sq_.slice(1,0,1) << std::endl;
    std::cout << "Num batches tracked: " << onlinenorm_module1->num_batches_tracked_.item<int64_t>() << std::endl;
    TORCH_CHECK(onlinenorm_module1->num_batches_tracked_.item<int64_t>() == 2, "Batch counter error.");


    // Evaluation pass (uses frozen online stats from pass 2)
    onlinenorm_module1->eval();
    torch::Tensor y1_eval = onlinenorm_module1->forward(x1); // Use original x1
    std::cout << "\nOutput y1_eval shape: " << y1_eval.sizes() << std::endl;
    std::cout << "y1_eval mean (channel 0, using frozen stats): " << y1_eval.select(1,0).mean().item<double>() << std::endl;
    std::cout << "y1_eval std (channel 0, using frozen stats): " << y1_eval.select(1,0).std(false).item<double>() << std::endl;
    // y1_eval's stats will depend on how different x1 is from the stats learned up to pass 2.

    TORCH_CHECK(y1_train_pass1.sizes() == x1.sizes(), "Output y1_train_pass1 shape mismatch!");


    // --- Test Case 2: No affine for g(x) ---
    std::cout << "\n--- Test Case 2: No affine_gn ---" << std::endl;
    OnlineNormalization onlinenorm_module2(num_features, 1e-5, 0.1, 0.1, /*affine_gn=*/false);
    onlinenorm_module2->train();
    torch::Tensor y2_train = onlinenorm_module2->forward(x1);
    std::cout << "y2_train mean (channel 0, no affine_gn): " << y2_train.select(1,0).mean().item<double>() << std::endl;
    TORCH_CHECK(onlinenorm_module2->alpha_.is_buffer(), "Alpha should be a buffer if not affine_gn");


    // --- Test Case 3: Check backward pass ---
    std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
    OnlineNormalization onlinenorm_module3(num_features, 1e-5, 0.1, 0.1, /*affine_gn=*/true);
    onlinenorm_module3->train();

    torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
    torch::Tensor y3 = onlinenorm_module3->forward(x3);
    torch::Tensor loss = y3.mean();
    loss.backward();

    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    bool grad_exists_alpha = onlinenorm_module3->alpha_.grad().defined() &&
                             onlinenorm_module3->alpha_.grad().abs().sum().item<double>() > 0;
    bool grad_exists_beta = onlinenorm_module3->beta_.grad().defined() &&
                            onlinenorm_module3->beta_.grad().abs().sum().item<double>() > 0;
    bool no_grad_online_mu = !onlinenorm_module3->online_mu_.grad().defined(); // Buffers should not have grads

    std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for alpha: " << (grad_exists_alpha ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for beta: " << (grad_exists_beta ? "true" : "false") << std::endl;
    std::cout << "No gradient for online_mu (buffer): " << (no_grad_online_mu ? "true" : "false") << std::endl;


    TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
    TORCH_CHECK(grad_exists_alpha, "No gradient for alpha!");
    TORCH_CHECK(grad_exists_beta, "No gradient for beta!");
    TORCH_CHECK(no_grad_online_mu, "online_mu buffer should not have gradient!");


    std::cout << "\nOnlineNormalization tests finished." << std::endl;
    return 0;
}





namespace xt::norm
{
    auto OnlineNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
