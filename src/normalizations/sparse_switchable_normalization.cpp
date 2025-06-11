#include "include/normalizations/sparse_switchable_normalization.h"


#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct SparseSwitchableNormImpl;

// The main module struct
struct SparseSwitchableNorm : torch::nn::ModuleHolder<SparseSwitchableNormImpl> {
    using torch::nn::ModuleHolder<SparseSwitchableNormImpl>::ModuleHolder;

    torch::Tensor forward(torch::Tensor x) {
        return impl_->forward(x);
    }
};

// The implementation struct
struct SparseSwitchableNormImpl : torch::nn::Module {
    int64_t num_features_;
    static const int kNumNormalizers = 3; // BN, IN, LN

    // BN parameters
    double eps_bn_;
    double momentum_bn_;
    bool affine_bn_;
    torch::Tensor running_mean_bn_;
    torch::Tensor running_var_bn_;
    torch::Tensor gamma_bn_;
    torch::Tensor beta_bn_;
    torch::Tensor num_batches_tracked_bn_;

    // IN parameters
    double eps_in_;
    bool affine_in_;
    torch::Tensor gamma_in_;
    torch::Tensor beta_in_;

    // LN parameters
    // LayerNorm normalizes over the last D dimensions. For typical (N,C,H,W),
    // and if we want to normalize over C,H,W, normalized_shape_ln_ would be {C,H,W}.
    // This makes LN params dependent on H,W.
    // A common simplification for SN is to have LN normalize only over C (like GroupNorm(1,C))
    // or to require fixed H,W for LN's specific affine params.
    // For this example, let's make LN normalize over C,H,W. We'll need to pass H,W or initialize LN later.
    // Or, use functional layer_norm and manage its affine params.
    // Let's make LN params dependent on H,W (initialized on first forward).
    std::vector<int64_t> normalized_shape_ln_; // e.g., {C,H,W}
    double eps_ln_;
    bool affine_ln_;
    torch::nn::LayerNorm layer_norm_{nullptr}; // To be initialized
    // Alternatively, manage gamma_ln, beta_ln manually if using functional layer_norm.
    // For simplicity with LayerNorm module, we defer its full init or assume it normalizes C.

    // Learnable importance scores (logits) for mixing normalizers
    // Shape: (num_features, kNumNormalizers) -> one set of scores per channel
    torch::Tensor mixing_logits_;

    SparseSwitchableNormImpl(int64_t num_features,
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

        // LN - Will be fully initialized in the first forward pass based on H, W
        // Or, if we fix LN to only normalize channels:
        // normalized_shape_ln_ = {num_features_}; // if input is (N, ..., C) and LN on C
        // if (affine_ln_) {
        //    layer_norm_ = torch::nn::LayerNorm(torch::nn::LayerNormOptions(normalized_shape_ln_).eps(eps_ln_).elementwise_affine(true));
        // } else {
        //    layer_norm_ = torch::nn::LayerNorm(torch::nn::LayerNormOptions(normalized_shape_ln_).eps(eps_ln_).elementwise_affine(false));
        // }
        // register_module("layer_norm", layer_norm_);
        // For this example, let's keep the dynamic LN initialization for (C,H,W) normalization.


        // Mixing logits
        // Initialize logits to be equal (e.g., zeros) for an initial uniform mixture.
        mixing_logits_ = register_parameter("mixing_logits", torch::zeros({num_features_, kNumNormalizers}));
    }


    torch::Tensor forward(torch::Tensor x) {
        // Input x expected to be 4D (N,C,H,W) for this implementation
        TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W). Got shape ", x.sizes());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Input channels mismatch. Expected ", num_features_, ", got ", x.size(1));

        int64_t N = x.size(0);
        int64_t C = x.size(1);
        int64_t H = x.size(2);
        int64_t W = x.size(3);

        std::vector<int64_t> bn_param_view_shape = {1, C, 1, 1}; // For broadcasting BN/IN affine params

        // --- Batch Normalization (y_bn) ---
        torch::Tensor y_bn;
        {
            torch::Tensor current_mean_bn, current_var_bn;
            std::vector<int64_t> reduce_dims_bn = {0, 2, 3}; // N, H, W

            if (this->is_training()) {
                current_mean_bn = x.mean(reduce_dims_bn, false);
                current_var_bn = (x - current_mean_bn.view(bn_param_view_shape)).pow(2).mean(reduce_dims_bn, false);
                running_mean_bn_ = (1.0 - momentum_bn_) * running_mean_bn_ + momentum_bn_ * current_mean_bn.detach();
                running_var_bn_  = (1.0 - momentum_bn_) * running_var_bn_  + momentum_bn_ * current_var_bn.detach();
                if (num_batches_tracked_bn_) num_batches_tracked_bn_ += 1;
            } else {
                current_mean_bn = running_mean_bn_;
                current_var_bn = running_var_bn_;
            }
            y_bn = (x - current_mean_bn.view(bn_param_view_shape)) /
                   torch::sqrt(current_var_bn.view(bn_param_view_shape) + eps_bn_);
            if (affine_bn_) {
                y_bn = y_bn * gamma_bn_.view(bn_param_view_shape) + beta_bn_.view(bn_param_view_shape);
            }
        }

        // --- Instance Normalization (y_in) ---
        torch::Tensor y_in;
        {
            std::vector<int64_t> reduce_dims_in = {2, 3}; // H, W
            auto mean_in = x.mean(reduce_dims_in, true);
            auto var_in = x.var(reduce_dims_in, false, true);
            y_in = (x - mean_in) / torch::sqrt(var_in + eps_in_);
            if (affine_in_) {
                y_in = y_in * gamma_in_.view(bn_param_view_shape) + beta_in_.view(bn_param_view_shape);
            }
        }

        // --- Layer Normalization (y_ln) ---
        // LN normalizes over C, H, W for each N.
        torch::Tensor y_ln;
        {
            if (!layer_norm_) { // Initialize LayerNorm on first forward pass
                normalized_shape_ln_ = {C, H, W};
                auto ln_options = torch::nn::LayerNormOptions(normalized_shape_ln_).eps(eps_ln_).elementwise_affine(affine_ln_);
                layer_norm_ = torch::nn::LayerNorm(ln_options);
                // If affine_ln_ is false, layer_norm_ won't have learnable weight/bias.
                // If true, they are created inside layer_norm_ module.
                // We need to register this dynamically created module if we want its params tracked.
                this->register_module("layer_norm_dynamic", layer_norm_);
                layer_norm_->to(x.device()); // Ensure it's on the same device
            }
             // Ensure the configured LN matches current input, if it was pre-configured differently
            if (layer_norm_->options.normalized_shape() != std::vector<int64_t>{C,H,W} && affine_ln_ ) {
                 TORCH_WARN("LayerNorm was configured for a different shape than current input. Re-initializing LN. This might lose learned LN parameters if H,W change.");
                 // Re-init (this is problematic for learned params if H,W change often)
                 normalized_shape_ln_ = {C, H, W};
                 auto ln_options = torch::nn::LayerNormOptions(normalized_shape_ln_).eps(eps_ln_).elementwise_affine(affine_ln_);
                 layer_norm_ = torch::nn::LayerNorm(ln_options);
                 this->register_module("layer_norm_dynamic", layer_norm_); // re-register
                 layer_norm_->to(x.device());
            } else if (layer_norm_->options.normalized_shape() != std::vector<int64_t>{C,H,W} && !affine_ln_) {
                 // if not affine, we can just use functional form or re-init without param loss
                 normalized_shape_ln_ = {C, H, W};
                 auto ln_options = torch::nn::LayerNormOptions(normalized_shape_ln_).eps(eps_ln_).elementwise_affine(affine_ln_);
                 layer_norm_ = torch::nn::LayerNorm(ln_options); // re-init non-affine LN
                 this->register_module("layer_norm_dynamic", layer_norm_);
                 layer_norm_->to(x.device());
            }

            y_ln = layer_norm_->forward(x);
        }

        // --- Calculate mixing weights ---
        // mixing_logits_ is (C, 3). Softmax over dim 1 (normalizers).
        torch::Tensor weights = torch::softmax(mixing_logits_, /*dim=*/1); // Shape (C, 3)

        // Reshape weights for broadcasting: (1, C, 1, 1) for each normalizer's weight
        torch::Tensor w_bn = weights.select(1, 0).view(bn_param_view_shape);
        torch::Tensor w_in = weights.select(1, 1).view(bn_param_view_shape);
        torch::Tensor w_ln = weights.select(1, 2).view(bn_param_view_shape);

        // --- Mix the outputs ---
        // The "sparsity" would come from these weights. If one weight is ~1 and others ~0 for a channel,
        // it effectively selects one normalizer for that channel.
        torch::Tensor output = w_bn * y_bn + w_in * y_in + w_ln * y_ln;

        return output;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "SparseSwitchableNorm(num_features=" << num_features_ << ", num_normalizers=" << kNumNormalizers
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
            stream << ", shape=dynamic";
        }
        stream << "))";
    }
};
TORCH_MODULE(SparseSwitchableNorm);


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    int64_t num_features = 32;
    int64_t N = 2, H = 8, W = 8; // Fixed H, W for easier LN testing

    // --- Test Case 1: SparseSwitchableNorm with defaults ---
    std::cout << "--- Test Case 1: SparseSwitchableNorm defaults ---" << std::endl;
    SparseSwitchableNorm ssn_module1(num_features);
    // std::cout << ssn_module1 << std::endl;
    std::cout << "Initial mixing_logits (first 2 features, all zeros): \n" << ssn_module1->mixing_logits_.slice(0,0,2) << std::endl;
    std::cout << "Initial mixing weights (softmax(zeros), first 2 features, ~0.33 each): \n"
              << torch::softmax(ssn_module1->mixing_logits_, 1).slice(0,0,2) << std::endl;

    torch::Tensor x1 = torch::randn({N, num_features, H, W});
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;

    // Training pass
    ssn_module1->train();
    torch::Tensor y1_train = ssn_module1->forward(x1);
    std::cout << "Output y1_train shape: " << y1_train.sizes() << std::endl;
    std::cout << "y1_train mean (all): " << y1_train.mean().item<double>() << std::endl;

    // Evaluation pass
    ssn_module1->eval();
    torch::Tensor y1_eval = ssn_module1->forward(x1);
    std::cout << "Output y1_eval shape: " << y1_eval.sizes() << std::endl;
    std::cout << "y1_eval mean (all): " << y1_eval.mean().item<double>() << std::endl;
    TORCH_CHECK(!torch::allclose(y1_train.mean(), y1_eval.mean()), "Train/Eval means should differ due to BN.");


    // --- Test Case 2: Forcing mixing weights to be "sparse" (favoring one normalizer) ---
    std::cout << "\n--- Test Case 2: Forcing sparse weights (favoring BN) ---" << std::endl;
    SparseSwitchableNorm ssn_module2(num_features, 1e-5,0.1,false, 1e-5,false, 1e-5,false); // All affine=false for clarity
    // Force logits for BN to be high, others low for feature 0
    ssn_module2->mixing_logits_.data()[0][0] = 10.0; // BN logit for feature 0
    ssn_module2->mixing_logits_.data()[0][1] = 0.0;  // IN logit
    ssn_module2->mixing_logits_.data()[0][2] = 0.0;  // LN logit
    std::cout << "Mixing_logits for feature 0: " << ssn_module2->mixing_logits_[0] << std::endl;
    auto weights_feat0 = torch::softmax(ssn_module2->mixing_logits_[0], 0);
    std::cout << "Resulting weights for feature 0 (should favor BN): " << weights_feat0 << std::endl;
    TORCH_CHECK(weights_feat0[0].item<float>() > 0.95, "Weights not strongly favoring BN for feature 0.");

    ssn_module2->eval(); // Use running stats for BN
    torch::Tensor y2 = ssn_module2->forward(x1);
    // y2's first channel should primarily reflect BN's output (mean 0, std 1 from running stats)
    std::cout << "Output y2 (feature 0 favored BN) mean: " << y2.select(1,0).mean().item<double>() << std::endl;
    std::cout << "Output y2 (feature 0 favored BN) std: " << y2.select(1,0).std(false).item<double>() << std::endl;


    // --- Test Case 3: Check backward pass ---
    std::cout << "\n--- Test Case 3: Backward pass check ---" << std::endl;
    SparseSwitchableNorm ssn_module3(num_features);
    ssn_module3->train();

    torch::Tensor x3 = torch::randn({N, num_features, H, W}, torch::requires_grad());
    torch::Tensor y3 = ssn_module3->forward(x3);
    torch::Tensor loss = y3.mean();
    loss.backward();

    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    bool grad_exists_logits = ssn_module3->mixing_logits_.grad().defined() &&
                              ssn_module3->mixing_logits_.grad().abs().sum().item<double>() > 0;
    bool grad_exists_gamma_bn = ssn_module3->gamma_bn_.grad().defined() &&
                                ssn_module3->gamma_bn_.grad().abs().sum().item<double>() > 0;
    // LayerNorm affine params are inside the layer_norm_ module if affine_ln=true
    bool grad_exists_gamma_ln = ssn_module3->layer_norm_ && ssn_module3->layer_norm_->named_parameters().contains("weight") &&
                                ssn_module3->layer_norm_->named_parameters()["weight"].grad().defined() &&
                                ssn_module3->layer_norm_->named_parameters()["weight"].grad().abs().sum().item<double>() > 0;


    std::cout << "Gradient exists for x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for mixing_logits: " << (grad_exists_logits ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for gamma_bn: " << (grad_exists_gamma_bn ? "true" : "false") << std::endl;
    std::cout << "Gradient exists for gamma_ln (if affine): " << (grad_exists_gamma_ln ? "true" : "false") << std::endl;

    TORCH_CHECK(grad_exists_x3, "No gradient for x3!");
    TORCH_CHECK(grad_exists_logits, "No gradient for mixing_logits!");
    TORCH_CHECK(grad_exists_gamma_bn, "No gradient for gamma_bn!");
    if (ssn_module3->affine_ln_) { // Only check if LN affine was true
        TORCH_CHECK(grad_exists_gamma_ln, "No gradient for LN gamma!");
    }

    // Print module structure after first forward (LN gets initialized)
    std::cout << "\nModule structure after initialization:" << std::endl;
    std::cout << ssn_module3 << std::endl;


    std::cout << "\nSparseSwitchableNorm tests finished." << std::endl;
    return 0;
}



namespace xt::norm
{
    auto SparseSwitchableNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
