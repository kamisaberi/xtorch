#include "include/normalizations/spade.h"


#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct SpadeImpl;

// The main module struct that users will interact with.
struct Spade : torch::nn::ModuleHolder<SpadeImpl> {
    using torch::nn::ModuleHolder<SpadeImpl>::ModuleHolder;

    // Forward method takes the main feature map x and the semantic segmentation map seg_map
    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& seg_map) {
        return impl_->forward(x, seg_map);
    }
};

// The implementation struct for SPADE
struct SpadeImpl : torch::nn::Module {
    int64_t norm_num_features_; // Number of features in input x (channels to be normalized)
    int64_t seg_map_channels_;  // Number of channels in the input segmentation map
    double eps_bn_;
    double momentum_bn_;

    // Batch Normalization components (without learnable affine parameters)
    torch::nn::BatchNorm2d batch_norm_{nullptr}; // Will be configured with affine=false

    // CNN for processing the segmentation map to produce gamma and beta
    // Usually a few conv layers. For simplicity, let's use two.
    // The number of hidden channels in this MLP can be a hyperparameter.
    int64_t hidden_channels_mlp_;
    torch::nn::Conv2d mlp_shared_conv1_{nullptr};
    // Separate conv layers for predicting gamma and beta
    torch::nn::Conv2d mlp_gamma_conv2_{nullptr};
    torch::nn::Conv2d mlp_beta_conv2_{nullptr};

    SpadeImpl(int64_t norm_num_features,
              int64_t seg_map_channels,
              int64_t hidden_channels_mlp = 128, // Common choice
              double eps_bn = 1e-5,
              double momentum_bn = 0.1)
        : norm_num_features_(norm_num_features),
          seg_map_channels_(seg_map_channels),
          hidden_channels_mlp_(hidden_channels_mlp),
          eps_bn_(eps_bn),
          momentum_bn_(momentum_bn) {

        TORCH_CHECK(norm_num_features_ > 0, "norm_num_features must be positive.");
        TORCH_CHECK(seg_map_channels_ > 0, "seg_map_channels must be positive.");
        TORCH_CHECK(hidden_channels_mlp_ > 0, "hidden_channels_mlp must be positive.");

        // 1. Initialize Batch Normalization (without affine transform)
        // The affine transform (gamma, beta) will come from the seg_map processing.
        batch_norm_ = torch::nn::BatchNorm2d(
            torch::nn::BatchNorm2dOptions(norm_num_features_).eps(eps_bn_).momentum(momentum_bn_).affine(false)
        );
        register_module("batch_norm", batch_norm_);

        // 2. Initialize the MLP (CNN) for processing the segmentation map
        // This MLP takes the seg_map (resized to x's H,W) and outputs gamma & beta maps.
        // Kernel size 3, padding 1 is common to preserve spatial dims.
        mlp_shared_conv1_ = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(seg_map_channels_, hidden_channels_mlp_, 3).padding(1)
        );
        register_module("mlp_shared_conv1", mlp_shared_conv1_);

        // Output layers for gamma and beta. Each has `norm_num_features_` output channels.
        mlp_gamma_conv2_ = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(hidden_channels_mlp_, norm_num_features_, 3).padding(1)
        );
        register_module("mlp_gamma_conv2", mlp_gamma_conv2_);

        mlp_beta_conv2_ = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(hidden_channels_mlp_, norm_num_features_, 3).padding(1)
        );
        register_module("mlp_beta_conv2", mlp_beta_conv2_);
    }

    torch::Tensor forward(const torch::Tensor& x, const torch::Tensor& seg_map_orig) {
        // x: input feature map (N, norm_C, Hx, Wx)
        // seg_map_orig: semantic segmentation map (N, seg_C, Hs, Ws)

        TORCH_CHECK(x.dim() == 4, "Input feature map x must be 4D (N, C, H, W). Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == norm_num_features_,
                    "Input x channels mismatch. Expected ", norm_num_features_, ", got ", x.size(1));
        TORCH_CHECK(seg_map_orig.dim() == 4, "Segmentation map must be 4D (N, C_seg, H, W). Got shape ", seg_map_orig.sizes());
        TORCH_CHECK(seg_map_orig.size(1) == seg_map_channels_,
                    "Segmentation map channels mismatch. Expected ", seg_map_channels_, ", got ", seg_map_orig.size(1));
        TORCH_CHECK(x.size(0) == seg_map_orig.size(0), "Batch sizes of x and seg_map must match.");

        // --- 1. Normalize x using Batch Normalization (no affine) ---
        torch::Tensor x_normalized = batch_norm_->forward(x); // (N, norm_C, Hx, Wx)

        // --- 2. Process segmentation map to get gamma and beta ---
        // Resize seg_map_orig to match spatial dimensions of x if they differ.
        torch::Tensor seg_map_processed = seg_map_orig;
        if (seg_map_orig.size(2) != x.size(2) || seg_map_orig.size(3) != x.size(3)) {
            seg_map_processed = torch::nn::functional::interpolate(
                seg_map_orig,
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{x.size(2), x.size(3)})
                    .mode(torch::kNearest) // or kBilinear, kNearest for discrete maps
            );
        }
        // seg_map_processed shape: (N, seg_C, Hx, Wx)

        // Pass through the MLP (CNN layers)
        torch::Tensor actv = mlp_shared_conv1_->forward(seg_map_processed);
        actv = torch::relu(actv); // Common activation

        torch::Tensor gamma_spatial = mlp_gamma_conv2_->forward(actv); // (N, norm_C, Hx, Wx)
        torch::Tensor beta_spatial  = mlp_beta_conv2_->forward(actv);  // (N, norm_C, Hx, Wx)

        // --- 3. Apply spatially-adaptive modulation ---
        // output = gamma_spatial * x_normalized + beta_spatial
        torch::Tensor output = x_normalized * gamma_spatial + beta_spatial;

        return output;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "Spade(norm_num_features=" << norm_num_features_
               << ", seg_map_channels=" << seg_map_channels_
               << ", hidden_channels_mlp=" << hidden_channels_mlp_
               << ", eps_bn=" << eps_bn_ << ", momentum_bn=" << momentum_bn_ << ")";
    }
};
TORCH_MODULE(Spade);


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    int64_t norm_C = 64;   // Channels in the main feature map x
    int64_t seg_C = 3;     // Channels in the segmentation map (e.g., one-hot encoded classes)
    int64_t hidden_mlp = 128;
    int64_t N = 2, Hx = 16, Wx = 16; // Dimensions for x
    int64_t Hs = 32, Ws = 32;       // Original dimensions for segmentation map (will be resized)

    // --- Test Case 1: SPADE with default parameters ---
    std::cout << "--- Test Case 1: SPADE defaults ---" << std::endl;
    Spade spade_module1(norm_C, seg_C, hidden_mlp);
    // std::cout << spade_module1 << std::endl;

    torch::Tensor x1 = torch::randn({N, norm_C, Hx, Wx});
    torch::Tensor seg_map1 = torch::randn({N, seg_C, Hs, Ws}); // Larger seg map
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;
    std::cout << "Input seg_map1 shape: " << seg_map1.sizes() << std::endl;

    // Training pass
    spade_module1->train();
    torch::Tensor y1_train = spade_module1->forward(x1, seg_map1);
    std::cout << "Output y1_train shape: " << y1_train.sizes() << std::endl;
    std::cout << "y1_train mean (all): " << y1_train.mean().item<double>() << std::endl;
    std::cout << "y1_train std (all): " << y1_train.std().item<double>() << std::endl;
    TORCH_CHECK(y1_train.sizes() == x1.sizes(), "Output y1_train shape mismatch!");

    // Evaluation pass
    spade_module1->eval();
    torch::Tensor y1_eval = spade_module1->forward(x1, seg_map1); // BN uses running stats
    std::cout << "Output y1_eval shape: " << y1_eval.sizes() << std::endl;
    std::cout << "y1_eval mean (all): " << y1_eval.mean().item<double>() << std::endl;
    TORCH_CHECK(!torch::allclose(y1_train.mean(), y1_eval.mean()),
                "Train and Eval output means should differ due to BN part.");


    // --- Test Case 2: Segmentation map already same size as x ---
    std::cout << "\n--- Test Case 2: Seg map same size as x ---" << std::endl;
    torch::Tensor seg_map2 = torch::randn({N, seg_C, Hx, Wx}); // Seg map same H, W as x
    spade_module1->eval(); // Keep in eval mode
    torch::Tensor y2_eval = spade_module1->forward(x1, seg_map2);
    std::cout << "Output y2_eval shape: " << y2_eval.sizes() << std::endl;
    TORCH_CHECK(y2_eval.sizes() == x1.sizes(), "Output y2_eval shape mismatch!");


    // --- Test Case 3: Check backward pass ---
    std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
    Spade spade_module3(norm_C, seg_C, hidden_mlp);
    spade_module3->train();

    torch::Tensor x3 = torch::randn({N, norm_C, Hx, Wx}, torch::requires_grad());
    torch::Tensor seg_map3 = torch::randn({N, seg_C, Hs, Ws}, torch::requires_grad()); // seg_map can also have grad

    torch::Tensor y3 = spade_module3->forward(x3, seg_map3);
    torch::Tensor loss = y3.mean();
    loss.backward();

    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    bool grad_exists_seg_map3 = seg_map3.grad().defined() && seg_map3.grad().abs().sum().item<double>() > 0;
    // Check grad for one of the MLP conv layers
    bool grad_exists_mlp_gamma_w = spade_module3->mlp_gamma_conv2_->weight.grad().defined() &&
                                   spade_module3->mlp_gamma_conv2_->weight.grad().abs().sum().item<double>() > 0;
    // Check grad for BN running stats (should not have grad, they are buffers)
    // BN's internal weight/bias are affine=false, so they won't exist or have grad.
    bool no_grad_bn_running_mean = !spade_module3->batch_norm_->named_buffers()["running_mean"].grad_fn().defined();


    std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for seg_map3: " << (grad_exists_seg_map3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for mlp_gamma_conv2.weight: " << (grad_exists_mlp_gamma_w ? "true" : "false") << std::endl;
    std::cout << "No gradient for BN running_mean (buffer): " << (no_grad_bn_running_mean ? "true" : "false") << std::endl;


    TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
    TORCH_CHECK(grad_exists_seg_map3, "No gradient for seg_map3!");
    TORCH_CHECK(grad_exists_mlp_gamma_w, "No gradient for mlp_gamma_conv2.weight!");
    TORCH_CHECK(no_grad_bn_running_mean, "BN running_mean should not have gradient!");


    std::cout << "\nSPADE tests finished." << std::endl;
    return 0;
}


namespace xt::norm
{
    auto SPADE::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
