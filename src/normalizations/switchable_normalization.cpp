#include "include/normalizations/switchable_normalization.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct SwitchableNormImpl;

// The main module struct
struct SwitchableNorm : torch::nn::ModuleHolder<SwitchableNormImpl> {
    using torch::nn::ModuleHolder<SwitchableNormImpl>::ModuleHolder;

    torch::Tensor forward(torch::Tensor x) {
        return impl_->forward(x);
    }
};

// The implementation struct
struct SwitchableNormImpl : torch::nn::Module {
    int64_t num_features_;
    static const int kNumNormalizers = 3; // BN, IN, LN

    // BN parameters
    double eps_bn_;
    double momentum_bn_;
    bool affine_bn_; // If true, BN has its own gamma_bn, beta_bn
    torch::Tensor running_mean_bn_;
    torch::Tensor running_var_bn_;
    torch::Tensor gamma_bn_;
    torch::Tensor beta_bn_;
    torch::Tensor num_batches_tracked_bn_;

    // IN parameters
    double eps_in_;
    bool affine_in_; // If true, IN has its own gamma_in, beta_in
    torch::Tensor gamma_in_;
    torch::Tensor beta_in_;

    // LN parameters
    std::vector<int64_t> normalized_shape_ln_; // e.g., {C,H,W}
    double eps_ln_;
    bool affine_ln_; // If true, LN has its own gamma_ln, beta_ln (managed by nn::LayerNorm module)
    torch::nn::LayerNorm layer_norm_{nullptr}; // To be initialized

    // Parameters for learning the switching weights (importance scores)
    // These are typically small linear layers operating on mean channel activations.
    // For simplicity, SN paper uses mean_weight and var_weight for BN, and simple params for IN/LN weights.
    // A common way: fc layers on global channel stats.
    // Let's use a simpler approach for scores: learnable per-channel logits directly.
    // This is similar to how `SparseSwitchableNorm` was implemented.
    // The original SN paper has a more complex scheme for deriving weights involving E[x] and Var[x] of current batch.
    // For simplicity, direct learnable logits per channel for BN, IN, LN.
    torch::Tensor mean_weight_logits_; // Logits for weights based on mean (often for BN)
    torch::Tensor var_weight_logits_;  // Logits for weights based on var (often for BN)
                                      // The paper's weight scheme is:
                                      // w_bn = softmax(lambda_mean_bn * E[x_ch] + lambda_var_bn * Var[x_ch])
                                      // w_in = softmax(lambda_in)
                                      // w_ln = softmax(lambda_ln)
                                      // This is quite involved.
                                      // A simpler variant (used in some implementations):
                                      // Learn 3 scores per channel, then softmax.
    torch::Tensor switching_logits_; // Shape: (num_features, kNumNormalizers)

    SwitchableNormImpl(int64_t num_features,
                       double eps_bn = 1e-5, double momentum_bn = 0.1, bool affine_bn = true,
                       double eps_in = 1e-5, bool affine_in = true,
                       double eps_ln = 1e-5, bool affine_ln = true)
        : num_features_(num_features),
          eps_bn_(eps_bn), momentum_bn_(momentum_bn), affine_bn_(affine_bn),
          eps_in_(eps_in), affine_in_(affine_in),
          eps_ln_(eps_ln), affine_ln_(affine_ln) {

        TORCH_CHECK(num_features > 0, "num_features must be positive.");

        // BN
        running_mean_bn_ = register_buffer("running_mean_bn", torch::zeros({num_features_}));
        running_var_bn_ = register_buffer("running_var_bn", torch::ones({num_features_}));
        num_batches_tracked_bn_ = register_buffer("num_batches_tracked_bn", torch::tensor(0, torch::kLong));
        if (affine_bn_) {
            gamma_bn_ = register_parameter("gamma_bn", torch::ones({num_features_}));
            beta_bn_ = register_parameter("beta_bn", torch::zeros({num_features_}));
        }

        // IN
        if (affine_in_) {
            gamma_in_ = register_parameter("gamma_in", torch::ones({num_features_}));
            beta_in_ = register_parameter("beta_in", torch::zeros({num_features_}));
        }

        // LN - Dynamically initialized in first forward pass for (C,H,W) normalization.

        // Switching logits: (num_features, 3) for BN, IN, LN scores per channel.
        // Initialized to zeros for equal initial weighting via softmax.
        switching_logits_ = register_parameter("switching_logits", torch::zeros({num_features_, kNumNormalizers}));
    }


    torch::Tensor forward(torch::Tensor x) {
        // Input x expected to be 4D (N,C,H,W)
        TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));

        int64_t N = x.size(0);
        int64_t C = x.size(1);
        int64_t H = x.size(2);
        int64_t W = x.size(3);

        std::vector<int64_t> affine_param_view_shape = {1, C, 1, 1}; // For broadcasting BN/IN affine params

        // --- Batch Normalization (y_bn) ---
        torch::Tensor y_bn;
        {
            torch::Tensor current_mean_bn, current_var_bn;
            std::vector<int64_t> reduce_dims_bn = {0, 2, 3}; // N, H, W

            if (this->is_training()) {
                current_mean_bn = x.mean(reduce_dims_bn, false);
                current_var_bn = (x - current_mean_bn.view(affine_param_view_shape)).pow(2).mean(reduce_dims_bn, false);
                running_mean_bn_ = (1.0 - momentum_bn_) * running_mean_bn_ + momentum_bn_ * current_mean_bn.detach();
                running_var_bn_  = (1.0 - momentum_bn_) * running_var_bn_  + momentum_bn_ * current_var_bn.detach();
                if (num_batches_tracked_bn_) num_batches_tracked_bn_ += 1;
            } else {
                current_mean_bn = running_mean_bn_;
                current_var_bn = running_var_bn_;
            }
            torch::Tensor x_bn_norm = (x - current_mean_bn.view(affine_param_view_shape)) /
                                      torch::sqrt(current_var_bn.view(affine_param_view_shape) + eps_bn_);
            if (affine_bn_) {
                y_bn = x_bn_norm * gamma_bn_.view(affine_param_view_shape) + beta_bn_.view(affine_param_view_shape);
            } else {
                y_bn = x_bn_norm;
            }
        }

        // --- Instance Normalization (y_in) ---
        torch::Tensor y_in;
        {
            std::vector<int64_t> reduce_dims_in = {2, 3}; // H, W
            auto mean_in = x.mean(reduce_dims_in, true);
            auto var_in = x.var(reduce_dims_in, false, true); // unbiased=false, keepdim=true
            torch::Tensor x_in_norm = (x - mean_in) / torch::sqrt(var_in + eps_in_);
            if (affine_in_) {
                y_in = x_in_norm * gamma_in_.view(affine_param_view_shape) + beta_in_.view(affine_param_view_shape);
            } else {
                y_in = x_in_norm;
            }
        }

        // --- Layer Normalization (y_ln) ---
        torch::Tensor y_ln;
        {
            // Initialize or ensure LayerNorm is correctly configured for current C, H, W
            if (!layer_norm_ || (layer_norm_ && layer_norm_->options.normalized_shape() != std::vector<int64_t>{C, H, W})) {
                if (layer_norm_ && (layer_norm_->options.normalized_shape() != std::vector<int64_t>{C,H,W})) {
                    TORCH_WARN_ONCE("Re-initializing LayerNorm due to shape change (C,H,W). This might reset learned LN affine parameters if H or W changed.");
                }
                normalized_shape_ln_ = {C, H, W};
                auto ln_options = torch::nn::LayerNormOptions(normalized_shape_ln_).eps(eps_ln_).elementwise_affine(affine_ln_);
                layer_norm_ = torch::nn::LayerNorm(ln_options);
                this->register_module("layer_norm_dynamic", layer_norm_); // Make sure it's registered
                layer_norm_->to(x.device());
            }
            y_ln = layer_norm_->forward(x); // If affine_ln=false, LN module applies normalization only.
                                           // If affine_ln=true, LN module also applies its internal gamma_ln, beta_ln.
        }

        // --- Calculate mixing weights ---
        // switching_logits_ is (C, 3). Softmax over dim 1 (normalizers).
        torch::Tensor weights = torch::softmax(switching_logits_, /*dim=*/1); // Shape (C, 3)

        // Reshape weights for broadcasting: (1, C, 1, 1) for each normalizer's weight
        torch::Tensor w_bn = weights.select(1, 0).view(affine_param_view_shape);
        torch::Tensor w_in = weights.select(1, 1).view(affine_param_view_shape);
        torch::Tensor w_ln = weights.select(1, 2).view(affine_param_view_shape);

        // --- Mix the outputs ---
        torch::Tensor output = w_bn * y_bn + w_in * y_in + w_ln * y_ln;

        return output;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "SwitchableNorm(num_features=" << num_features_ << ", num_normalizers=" << kNumNormalizers
               << "\n  BN(eps=" << eps_bn_ << ", mom=" << momentum_bn_ << ", aff=" << affine_bn_ << ")"
               << "\n  IN(eps=" << eps_in_ << ", aff=" << affine_in_ << ")"
               << "\n  LN(eps=" << eps_ln_ << ", aff=" << affine_ln_;
        if (layer_norm_) {
             stream << ", shape=[";
            for(size_t i=0; i<layer_norm_->options.normalized_shape().size(); ++i) {
                stream << layer_norm_->options.normalized_shape()[i] << (i == layer_norm_->options.normalized_shape().size()-1 ? "" : ", ");
            }
            stream << "]";
        } else {
            stream << ", shape=dynamic_on_first_fwd";
        }
        stream << "))";
    }
};
TORCH_MODULE(SwitchableNorm);


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    int64_t num_features = 32;
    int64_t N = 2, H = 8, W = 8; // Example dimensions, H,W are fixed for this test

    // --- Test Case 1: SwitchableNorm with defaults ---
    std::cout << "--- Test Case 1: SwitchableNorm defaults ---" << std::endl;
    SwitchableNorm sn_module1(num_features);
    // std::cout << sn_module1 << std::endl; // Will print detailed after first forward for LN
    std::cout << "Initial switching_logits (first 2 features, all zeros): \n" << sn_module1->switching_logits_.slice(0,0,2) << std::endl;
    auto initial_weights = torch::softmax(sn_module1->switching_logits_, 1).slice(0,0,2);
    std::cout << "Initial mixing weights (first 2 features, ~0.33 each): \n" << initial_weights << std::endl;


    torch::Tensor x1 = torch::randn({N, num_features, H, W});
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;

    // Training pass
    sn_module1->train();
    torch::Tensor y1_train = sn_module1->forward(x1);
    std::cout << "Output y1_train shape: " << y1_train.sizes() << std::endl;
    std::cout << "y1_train mean (all): " << y1_train.mean().item<double>() << std::endl;

    // Evaluation pass
    sn_module1->eval();
    torch::Tensor y1_eval = sn_module1->forward(x1);
    std::cout << "Output y1_eval shape: " << y1_eval.sizes() << std::endl;
    std::cout << "y1_eval mean (all): " << y1_eval.mean().item<double>() << std::endl;
    TORCH_CHECK(!torch::allclose(y1_train.mean(), y1_eval.mean(), 1e-5, 1e-5), // Expect difference
                "Train and Eval output means should typically differ due to BN part.");


    // --- Test Case 2: Forcing mixing weights to favor one normalizer (e.g., IN) ---
    std::cout << "\n--- Test Case 2: Forcing weights (favoring IN) ---" << std::endl;
    // Disable affine for sub-normalizers to isolate effect of mixing normalized outputs
    SwitchableNorm sn_module2(num_features, 1e-5,0.1,false, 1e-5,false, 1e-5,false);
    // Force logits for IN to be high, others low for all features
    sn_module2->switching_logits_.data().select(1, 0).fill_(-10.0); // BN logit
    sn_module2->switching_logits_.data().select(1, 1).fill_(10.0);  // IN logit
    sn_module2->switching_logits_.data().select(1, 2).fill_(-10.0); // LN logit

    auto forced_weights_ch0 = torch::softmax(sn_module2->switching_logits_[0], 0);
    std::cout << "Forced weights for feature 0 (should favor IN): " << forced_weights_ch0 << std::endl;
    TORCH_CHECK(forced_weights_ch0[1].item<float>() > 0.99, "Weights not strongly favoring IN.");

    sn_module2->eval(); // Mode does not affect IN or LN core stats if no BN running stats used.
    torch::Tensor y2 = sn_module2->forward(x1); // x1 from previous test
    // y2 should primarily reflect IN's output (mean ~0, std ~1 per instance per channel)
    std::cout << "Output y2 (favored IN) mean (instance 0, channel 0): "
              << y2.select(0,0).select(0,0).mean().item<double>() << std::endl;
    std::cout << "Output y2 (favored IN) std (instance 0, channel 0): "
              << y2.select(0,0).select(0,0).std(false).item<double>() << std::endl;
    TORCH_CHECK(std::abs(y2.select(0,0).select(0,0).mean().item<double>()) < 1e-1, "Mean for IN-favored not close to 0.");


    // --- Test Case 3: Check backward pass ---
    std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
    SwitchableNorm sn_module3(num_features, 1e-5,0.1,true, 1e-5,true, 1e-5,true); // All affine true
    sn_module3->train();

    torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
    // First forward to initialize LN
    torch::Tensor y3_init_ln = sn_module3->forward(x3.detach());


    // Second forward for backward
    torch::Tensor y3 = sn_module3->forward(x3);
    torch::Tensor loss = y3.mean();
    loss.backward();

    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    bool grad_exists_logits = sn_module3->switching_logits_.grad().defined() &&
                              sn_module3->switching_logits_.grad().abs().sum().item<double>() > 0;
    bool grad_exists_gamma_bn = sn_module3->gamma_bn_.grad().defined() &&
                                sn_module3->gamma_bn_.grad().abs().sum().item<double>() > 0;
    bool grad_exists_gamma_in = sn_module3->gamma_in_.grad().defined() &&
                                sn_module3->gamma_in_.grad().abs().sum().item<double>() > 0;
    // LayerNorm affine params are inside the layer_norm_ module
    bool grad_exists_gamma_ln = sn_module3->layer_norm_ && sn_module3->layer_norm_->named_parameters().contains("weight") &&
                                sn_module3->layer_norm_->named_parameters()["weight"].grad().defined() &&
                                sn_module3->layer_norm_->named_parameters()["weight"].grad().abs().sum().item<double>() > 0;

    std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for switching_logits: " << (grad_exists_logits ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for gamma_bn: " << (grad_exists_gamma_bn ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for gamma_in: " << (grad_exists_gamma_in ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for gamma_ln: " << (grad_exists_gamma_ln ? "true" : "false") << std::endl;


    TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
    TORCH_CHECK(grad_exists_logits, "No gradient for switching_logits!");
    TORCH_CHECK(grad_exists_gamma_bn, "No gradient for gamma_bn!");
    TORCH_CHECK(grad_exists_gamma_in, "No gradient for gamma_in!");
    TORCH_CHECK(grad_exists_gamma_ln, "No gradient for LN gamma!");

    // Print module structure after LN initialization
    std::cout << "\nModule structure after LN initialization:" << std::endl;
    std::cout << sn_module3 << std::endl;


    std::cout << "\nSwitchableNorm tests finished." << std::endl;
    return 0;
}





namespace xt::norm
{
    auto SwitchableNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
