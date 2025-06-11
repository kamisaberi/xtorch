#include "include/normalizations/conditional_instance_normalization.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct ConditionalInstanceNormalizationImpl;

// The main module struct that users will interact with.
struct ConditionalInstanceNormalization : torch::nn::ModuleHolder<ConditionalInstanceNormalizationImpl> {
    using torch::nn::ModuleHolder<ConditionalInstanceNormalizationImpl>::ModuleHolder;

    // Forward method takes the main input x and the conditioning input
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& conditioning_input) {
        return impl_->forward(x, conditioning_input);
    }
};

// The implementation struct
struct ConditionalInstanceNormalizationImpl : torch::nn::Module {
    int64_t num_features_;          // Number of features in input x (channels)
    int64_t cond_embedding_dim_;    // Dimensionality of the conditioning input vector
    double eps_;
    int64_t cond_hidden_dim_;       // Hidden dimension for the conditioning network

    // Conditioning network layers (to produce gamma and beta)
    torch::nn::Linear fc_cond1_{nullptr}; // Optional first layer
    torch::nn::Linear fc_cond_out_{nullptr}; // Output layer for gamma and beta

    ConditionalInstanceNormalizationImpl(int64_t num_features,
                                         int64_t cond_embedding_dim,
                                         int64_t cond_hidden_dim = 0, // 0 means no hidden layer for cond net
                                         double eps = 1e-5)
        : num_features_(num_features),
          cond_embedding_dim_(cond_embedding_dim),
          cond_hidden_dim_(cond_hidden_dim),
          eps_(eps) {

        TORCH_CHECK(num_features > 0, "num_features must be positive.");
        TORCH_CHECK(cond_embedding_dim > 0, "cond_embedding_dim must be positive.");

        // Instance Normalization does not have running_mean/var in the same way BN does.
        // Statistics are computed per instance.

        // Setup conditioning network
        if (cond_hidden_dim_ <= 0) { // Direct mapping from conditioning_input to gamma/beta
            fc_cond_out_ = torch::nn::Linear(cond_embedding_dim_, 2 * num_features_); // 2 for gamma and beta
            register_module("fc_cond_out", fc_cond_out_);
        } else {
            fc_cond1_ = torch::nn::Linear(cond_embedding_dim_, cond_hidden_dim_);
            fc_cond_out_ = torch::nn::Linear(cond_hidden_dim_, 2 * num_features_);
            register_module("fc_cond1", fc_cond1_);
            register_module("fc_cond_out", fc_cond_out_);
        }
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& conditioning_input) {
        // x: input tensor (N, C, D1, D2, ...) where C is num_features_
        // conditioning_input: (N, cond_embedding_dim_)

        TORCH_CHECK(x.dim() >= 2, "Input tensor x must have at least 2 dimensions (N, C, ...). Got ", x.dim());
        TORCH_CHECK(x.size(0) == conditioning_input.size(0),
                    "Batch size of x (", x.size(0), ") and conditioning_input (", conditioning_input.size(0), ") must match.");
        TORCH_CHECK(x.size(1) == num_features_,
                    "Number of input features (channels) in x mismatch. Expected ", num_features_,
                    ", but got ", x.size(1), " for input x of shape ", x.sizes());
        TORCH_CHECK(conditioning_input.dim() == 2 && conditioning_input.size(1) == cond_embedding_dim_,
                    "Conditioning input must be 2D with shape (N, cond_embedding_dim). Got shape: ", conditioning_input.sizes());


        // --- 1. Instance Normalization part ---
        torch::Tensor x_normalized;

        if (x.dim() > 2) { // Input has spatial/sequential dimensions (N, C, D1, ...)
            std::vector<int64_t> reduce_dims_for_stats; // Dims to average over for mean/var (D1, D2, ...)
            for (int64_t i = 2; i < x.dim(); ++i) {
                reduce_dims_for_stats.push_back(i);
            }
            // keepdim=true for broadcasting
            auto mean = x.mean(reduce_dims_for_stats, /*keepdim=*/true);
            auto var = x.var(reduce_dims_for_stats, /*unbiased=*/false, /*keepdim=*/true);
            x_normalized = (x - mean) / torch::sqrt(var + eps_);
        } else { // Input is 2D (N, C). InstanceNorm on a single point per channel results in 0.
            x_normalized = torch::zeros_like(x);
        }

        // --- 2. Generate Gamma and Beta from conditioning_input ---
        torch::Tensor cond_features = conditioning_input;
        if (fc_cond1_) { // If hidden layer exists
            cond_features = fc_cond1_->forward(cond_features);
            cond_features = torch::relu(cond_features); // Common activation
        }
        torch::Tensor gamma_beta_params = fc_cond_out_->forward(cond_features); // (N, 2 * num_features_)

        auto chunks = torch::chunk(gamma_beta_params, 2, /*dim=*/1);
        torch::Tensor gamma_generated = chunks[0]; // (N, num_features_)
        torch::Tensor beta_generated  = chunks[1]; // (N, num_features_)

        // --- 3. Reshape generated Gamma and Beta for broadcasting ---
        // Desired shape: (N, C, 1, 1, ...) to match x_normalized (N, C, D1, D2, ...)
        std::vector<int64_t> affine_param_view_shape;
        affine_param_view_shape.push_back(x.size(0));      // N
        affine_param_view_shape.push_back(num_features_);  // C
        for (int64_t i = 2; i < x.dim(); ++i) {
            affine_param_view_shape.push_back(1);
        }

        gamma_generated = gamma_generated.view(affine_param_view_shape);
        beta_generated  = beta_generated.view(affine_param_view_shape);

        // --- 4. Apply conditional affine transformation ---
        return gamma_generated * x_normalized + beta_generated;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "ConditionalInstanceNormalization(num_features=" << num_features_
               << ", cond_embedding_dim=" << cond_embedding_dim_
               << ", cond_hidden_dim=" << (fc_cond1_ ? std::to_string(cond_hidden_dim_) : "0 (direct)")
               << ", eps=" << eps_ << ")";
    }
};
TORCH_MODULE(ConditionalInstanceNormalization);


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    int64_t num_features = 3;       // Channels in x
    int64_t cond_embedding_dim = 10; // Dimension of conditioning vector
    int64_t N = 4;                  // Batch size

    // --- Test Case 1: 4D input x (NCHW), no hidden layer in conditioning net ---
    std::cout << "--- Test Case 1: 4D input x (NCHW), no hidden layer in cond_net ---" << std::endl;
    ConditionalInstanceNormalization cin_module1(num_features, cond_embedding_dim, /*cond_hidden_dim=*/0);
    // std::cout << cin_module1 << std::endl;

    torch::Tensor x1 = torch::randn({N, num_features, 8, 8});
    torch::Tensor cond1 = torch::randn({N, cond_embedding_dim});
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
    std::cout << "Cond cond1 shape: " << cond1.sizes() << std::endl;

    // InstanceNorm doesn't have train/eval mode differences in its core logic
    cin_module1->eval(); // or train(), behavior for IN core is the same
    torch::Tensor y1 = cin_module1->forward(x1, cond1);
    std::cout << "Output y1 shape: " << y1.sizes() << std::endl;

    // For each instance n and channel c, mean(y1[n,c,:,:]) should be approx. beta_generated[n,c]
    // and std(y1[n,c,:,:]) should be approx. gamma_generated[n,c]
    // Let's check for the first instance, first channel
    auto y1_inst0_ch0 = y1.select(0,0).select(0,0); // N=0, C=0
    std::cout << "y1 [0,0,:,:] mean: " << y1_inst0_ch0.mean().item<double>() << std::endl;
    std::cout << "y1 [0,0,:,:] std:  " << y1_inst0_ch0.std(false).item<double>() << std::endl;


    // --- Test Case 2: 2D input x (NC), with hidden layer in conditioning net ---
    std::cout << "\n--- Test Case 2: 2D input x (NC), with hidden layer in cond_net ---" << std::endl;
    int64_t cond_hidden = 32;
    ConditionalInstanceNormalization cin_module2(num_features, cond_embedding_dim, cond_hidden);
    // std::cout << cin_module2 << std::endl;

    torch::Tensor x2 = torch::randn({N, num_features}); // e.g. (4, 3)
    torch::Tensor cond2 = torch::randn({N, cond_embedding_dim});
    std::cout << "Input x2 shape: " << x2.sizes() << std::endl;
    std::cout << "Cond cond2 shape: " << cond2.sizes() << std::endl;

    torch::Tensor y2 = cin_module2->forward(x2, cond2);
    std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
    // For 2D input (N,C), x_normalized will be all zeros.
    // So, y2 should be equal to the generated beta parameters (reshaped).
    std::cout << "y2[0] (should be beta_generated[0]): " << y2[0] << std::endl;


    // --- Test Case 3: Check backward pass (requires gradients) ---
    std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
    ConditionalInstanceNormalization cin_module3(num_features, cond_embedding_dim, cond_hidden);
    cin_module3->train(); // Ensure parameters of conditioning net have requires_grad=true

    torch::Tensor x3 = torch::randn({N, num_features, 6, 6}, torch::requires_grad());
    torch::Tensor cond3 = torch::randn({N, cond_embedding_dim}, torch::requires_grad());

    torch::Tensor y3 = cin_module3->forward(x3, cond3);
    torch::Tensor loss = y3.mean();
    loss.backward();

    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    bool grad_exists_cond3 = cond3.grad().defined() && cond3.grad().abs().sum().item<double>() > 0;
    bool grad_exists_fc_out_weight = cin_module3->fc_cond_out_->weight.grad().defined() &&
                                     cin_module3->fc_cond_out_->weight.grad().abs().sum().item<double>() > 0;

    std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for cond3: " << (grad_exists_cond3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for fc_cond_out.weight: " << (grad_exists_fc_out_weight ? "true" : "false") << std::endl;

    TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
    TORCH_CHECK(grad_exists_cond3, "No gradient for cond3!");
    TORCH_CHECK(grad_exists_fc_out_weight, "No gradient for fc_cond_out.weight!");

    std::cout << "\nConditionalInstanceNormalization tests finished." << std::endl;
    return 0;
}




namespace xt::norm
{
    auto ConditionalInstanceNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
