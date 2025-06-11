#include "include/normalizations/weight_normalization.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <cmath> // For std::sqrt

// Forward declaration for the Impl struct
struct WeightNormLinearImpl;

// The main module struct that users will interact with.
// This module acts like nn::Linear but with weight normalization.
struct WeightNormLinear : torch::nn::ModuleHolder<WeightNormLinearImpl> {
    WeightNormLinear(int64_t in_features, int64_t out_features, bool bias = true, int64_t norm_dim = 0);

    torch::Tensor forward(torch::Tensor x);

    // Optional: Method to remove weight normalization and revert to standard weights
    void remove_weight_norm();
};

// The implementation struct for WeightNormLinear
struct WeightNormLinearImpl : torch::nn::Module {
    int64_t in_features_;
    int64_t out_features_;
    bool bias_defined_;
    int64_t norm_dim_; // Dimension along which to compute the norm of 'v'
                       // For nn.Linear, weight is (out_features, in_features).
                       // norm_dim = 0: normalize each row (each output filter's connection to inputs).
                       // norm_dim = 1: normalize each col (each input's connection to all outputs).
                       // Default in PyTorch's weight_norm is 0.

    // Learnable parameters: g (magnitude/gain) and v (direction)
    // v_param_ stores the unnormalized direction vectors.
    // g_param_ stores the gain scalars/vectors.
    torch::Tensor v_param_; // Shape: (out_features, in_features)
    torch::Tensor g_param_; // Shape: (out_features, 1) if norm_dim=0, or scalar, or (1,in_features) if norm_dim=1

    // Optional learnable bias
    torch::Tensor bias_param_;

    // Epsilon for norm calculation
    double eps_ = 1e-12; // From PyTorch's implementation

public:
    WeightNormLinearImpl(int64_t in_features, int64_t out_features, bool bias = true, int64_t norm_dim = 0)
        : in_features_(in_features),
          out_features_(out_features),
          bias_defined_(bias),
          norm_dim_(norm_dim) {

        TORCH_CHECK(in_features_ > 0, "in_features must be positive.");
        TORCH_CHECK(out_features_ > 0, "out_features must be positive.");
        TORCH_CHECK(norm_dim_ == 0 || norm_dim_ == 1, "norm_dim must be 0 or 1 for Linear layer weights.");

        // Initialize v_param_ (unnormalized direction)
        // Standard nn.Linear initialization often uses kaiming_uniform_
        v_param_ = register_parameter("v", torch::randn({out_features_, in_features_}));
        torch::nn::init::kaiming_uniform_(v_param_, std::sqrt(5)); // Matches default Linear init more closely

        // Initialize g_param_ (gain)
        // g is typically initialized based on the initial norm of v.
        // If W = g * (v/||v||), then ||W|| = g.
        // We want g to be such that initially, W has a similar scale to standard initialized weights.
        // PyTorch's weight_norm initializes g as the norm of v along norm_dim.
        torch::Tensor v_norm = v_param_.norm(2, norm_dim_, true); // keepdim=true
        g_param_ = register_parameter("g", v_norm.clone().detach()); // Initialize g with ||v||

        if (bias_defined_) {
            bias_param_ = register_parameter("bias", torch::zeros({out_features_}));
            // Match Linear's bias init if possible (fan_in for kaiming_uniform for weights)
            if (v_param_.defined()) { // v_param should always be defined here
                double fan_in = v_param_.size(1);
                double bound = 1.0 / std::sqrt(fan_in);
                 if (linear_bias_requires_grad()) // Check if bias requires grad if it's a parameter
                    torch::nn::init::uniform_(bias_param_, -bound, bound);
            }
        }
    }

    // Computes the effective weight W = g * (v / ||v||)
    torch::Tensor compute_weight() {
        torch::Tensor v_norm = v_param_.norm(2, norm_dim_, true); // (out_features, 1) or (1, in_features)
        torch::Tensor v_normalized = v_param_ / (v_norm + eps_);

        // g_param_ shape must match v_norm for broadcasting here
        // If norm_dim=0, v_norm is (out_features, 1), g_param_ should be (out_features, 1)
        // If norm_dim=1, v_norm is (1, in_features), g_param_ should be (1, in_features)
        // Our g_param_ is initialized with shape of v_norm.
        torch::Tensor weight = g_param_ * v_normalized;
        return weight;
    }

    torch::Tensor forward_impl(torch::Tensor x) {
        // x: input tensor of shape (Batch, ..., in_features_)
        // Output shape: (Batch, ..., out_features_)

        TORCH_CHECK(x.size(-1) == in_features_,
                    "Last dimension of input x (", x.size(-1), ") must match in_features (", in_features_, ").");

        torch::Tensor weight = compute_weight(); // Get the reparameterized weight
        return torch::linear(x, weight, bias_param_);
    }

    void remove_weight_norm_impl() {
        // To remove weight normalization, we compute the effective weight W
        // and then re-register it as the 'weight' parameter, deleting 'g' and 'v'.
        // This is non-trivial to do perfectly in C++ with Module's parameter system
        // without careful unregistering and reregistering.
        // PyTorch's Python version uses delattr and setattr.

        // For this example, we'll compute W and store it.
        // A true 'remove' would modify the parameter list of this module.
        // This requires more advanced manipulation of the _parameters map.

        // Get current effective weight
        torch::Tensor current_W;
        {
            torch::NoGradGuard no_grad; // Don't track gradients for this computation
            current_W = compute_weight().clone().detach();
        }

        // This is where you would ideally:
        // 1. Unregister 'g' and 'v' parameters.
        // 2. Register 'weight' as a new parameter with value current_W.
        // _parameters.erase("g"); _parameters.erase("v");
        // register_parameter("weight", current_W);
        // And then make forward use this new 'weight' parameter.
        // For simplicity, this example won't fully reconfigure the module.
        // It would require changing the class structure significantly or advanced parameter map manipulation.

        // For now, let's just replace v_param and g_param such that compute_weight() returns current_W
        // effectively freezing W. This isn't a true removal.
        // A true removal makes it a standard Linear layer.
        // This is a placeholder for the complex operation.
        // A more robust way: have a flag, and if removed, forward uses a stored 'weight_removed_'.
        TORCH_WARN("remove_weight_norm() is a complex operation to fully implement like PyTorch's Python version."
                   " This C++ example's 'remove' is conceptual or would require significant module restructuring.");
        // As a simple simulation of removal for future forward calls if compute_weight were more complex:
        // this->v_param_.set_requires_grad(false);
        // this->g_param_.set_requires_grad(false);
        // // Or, more drastically but not quite right:
        // // this->v_param_.data().copy_(current_W);
        // // this->g_param_.data().fill_(1.0); // If compute_weight becomes v * (g/||v||)
        // // This is NOT a proper removal.
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "WeightNormLinear(in_features=" << in_features_
               << ", out_features=" << out_features_
               << ", bias=" << (bias_defined_ ? "true" : "false")
               << ", norm_dim=" << norm_dim_ << ")";
    }
};


// Define public methods for the ModuleHolder
WeightNormLinear::WeightNormLinear(int64_t in_features, int64_t out_features, bool bias, int64_t norm_dim)
    : ModuleHolder(std::make_shared<WeightNormLinearImpl>(in_features, out_features, bias, norm_dim)) {}

torch::Tensor WeightNormLinear::forward(torch::Tensor x) {
    return impl_->forward_impl(x);
}

void WeightNormLinear::remove_weight_norm() {
    impl_->remove_weight_norm_impl();
}


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    int64_t in_dim = 10;
    int64_t out_dim = 5;

    // --- Test Case 1: WeightNormLinear basic functionality ---
    std::cout << "--- Test Case 1: WeightNormLinear defaults ---" << std::endl;
    WeightNormLinear wn_linear1(in_dim, out_dim); // norm_dim = 0 by default
    // std::cout << wn_linear1 << std::endl;

    torch::Tensor x1 = torch::randn({4, in_dim}); // Batch of 4
    std::cout << "Input x1 shape: " << x1.sizes() << std::endl;

    // Check initial g and v parameters
    std::cout << "Initial g_param (shape " << wn_linear1->impl_->g_param_.sizes() << "): \n"
              << wn_linear1->impl_->g_param_.slice(0, 0, 2) << std::endl; // First 2 rows
    std::cout << "Initial v_param norm (row 0): "
              << wn_linear1->impl_->v_param_[0].norm().item<double>() << std::endl;
    // g_param for row 0 should be approx norm of v_param row 0.
    TORCH_CHECK(torch::allclose(wn_linear1->impl_->g_param_[0][0], wn_linear1->impl_->v_param_[0].norm()),
               "g_param initialization failed for row 0");


    torch::Tensor y1 = wn_linear1->forward(x1);
    std::cout << "Output y1 shape: " << y1.sizes() << std::endl;
    TORCH_CHECK(y1.size(0) == 4 && y1.size(1) == out_dim, "Output y1 shape mismatch!");

    // Check effective weight norm
    torch::Tensor effective_w1 = wn_linear1->impl_->compute_weight();
    // For norm_dim=0, each row of effective_w1 should have norm equal to corresponding g_param element.
    torch::Tensor row0_w1_norm = effective_w1[0].norm();
    torch::Tensor g0_val = wn_linear1->impl_->g_param_[0][0]; // g_param is (out_features, 1)
    std::cout << "Norm of effective_w1 row 0: " << row0_w1_norm.item<double>()
              << ", g_param[0][0]: " << g0_val.item<double>() << std::endl;
    TORCH_CHECK(torch::allclose(row0_w1_norm, g0_val), "Effective weight row norm mismatch with g_param.");


    // --- Test Case 2: norm_dim = 1 ---
    std::cout << "\n--- Test Case 2: norm_dim = 1 ---" << std::endl;
    WeightNormLinear wn_linear2(in_dim, out_dim, true, /*norm_dim=*/1);
    // std::cout << wn_linear2 << std::endl;
    std::cout << "Initial g_param (shape " << wn_linear2->impl_->g_param_.sizes() << "): \n"
              << wn_linear2->impl_->g_param_.slice(1, 0, 2) << std::endl; // First 2 cols (if g is 1,in_features)
    // Check g_param for column 0, should be approx norm of v_param column 0.
    torch::Tensor v_col0_norm = wn_linear2->impl_->v_param_.slice(1,0,1).norm();
    TORCH_CHECK(torch::allclose(wn_linear2->impl_->g_param_[0][0], wn_linear2->impl_->v_param_.select(1,0).norm()),
                "g_param initialization failed for col 0 with norm_dim=1");

    torch::Tensor y2 = wn_linear2->forward(x1); // Use same x1
    std::cout << "Output y2 shape: " << y2.sizes() << std::endl;
    torch::Tensor effective_w2 = wn_linear2->impl_->compute_weight();
    // For norm_dim=1, each col of effective_w2 should have norm equal to corresponding g_param element.
    torch::Tensor col0_w2_norm = effective_w2.select(1,0).norm(); // Norm of first column
    torch::Tensor g_col0_val = wn_linear2->impl_->g_param_[0][0]; // g_param is (1, in_features)
    std::cout << "Norm of effective_w2 col 0: " << col0_w2_norm.item<double>()
              << ", g_param[0][0]: " << g_col0_val.item<double>() << std::endl;
    TORCH_CHECK(torch::allclose(col0_w2_norm, g_col0_val), "Effective weight col norm mismatch with g_param for norm_dim=1.");


    // --- Test Case 3: Check backward pass and parameter updates ---
    std::cout << "\n--- Test Case 3: Backward pass and parameter updates ---" << std::endl;
    WeightNormLinear wn_linear3(in_dim, out_dim);
    wn_linear3->train();

    // Access g and v parameters
    torch::Tensor& g_param_ref = wn_linear3->impl_->g_param_;
    torch::Tensor& v_param_ref = wn_linear3->impl_->v_param_;
    double initial_g00 = g_param_ref[0][0].item<double>();
    double initial_v00 = v_param_ref[0][0].item<double>();

    torch::optim::SGD optimizer(wn_linear3->parameters(), /*lr=*/0.1);

    optimizer.zero_grad();
    torch::Tensor x3 = torch::randn({2, in_dim}, torch::requires_grad());
    torch::Tensor y3 = wn_linear3->forward(x3);
    torch::Tensor loss = y3.mean();
    loss.backward();
    optimizer.step();

    std::cout << "Initial g_param[0][0]: " << initial_g00 << ", Updated: " << g_param_ref[0][0].item<double>() << std::endl;
    std::cout << "Initial v_param[0][0]: " << initial_v00 << ", Updated: " << v_param_ref[0][0].item<double>() << std::endl;
    TORCH_CHECK(g_param_ref[0][0].item<double>() != initial_g00, "g_param did not update.");
    TORCH_CHECK(v_param_ref[0][0].item<double>() != initial_v00, "v_param did not update.");

    bool grad_exists_x3 = x3.grad().defined() && x3.grad().abs().sum().item<double>() > 0;
    std::cout << "Gradient exists for input x3: " << (grad_exists_x3 ? "true" : "false") << std::endl;
    TORCH_CHECK(grad_exists_x3, "Input x3 did not receive gradient!");


    // --- Test Case 4: Remove weight norm (conceptual check) ---
    std::cout << "\n--- Test Case 4: remove_weight_norm (conceptual) ---" << std::endl;
    WeightNormLinear wn_linear4(in_dim, out_dim);
    wn_linear4->remove_weight_norm(); // Calls the placeholder method
    // Further checks would require actually changing module structure or having a flag.
    std::cout << "remove_weight_norm called. (Functionality is placeholder in this example)" << std::endl;


    std::cout << "\nWeightNormLinear tests finished." << std::endl;
    return 0;
}



namespace xt::norm
{
    auto WeightNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
