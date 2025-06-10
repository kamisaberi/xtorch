#include "include/normalizations/batch_channel_normalization.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <numeric> // For std::iota

// Forward declaration for the Impl struct
struct BatchChannelNormalizationImpl;

// The main module struct that users will interact with.
struct BatchChannelNormalization : torch::nn::ModuleHolder<BatchChannelNormalizationImpl> {
    using torch::nn::ModuleHolder<BatchChannelNormalizationImpl>::ModuleHolder;

    torch::Tensor forward(torch::Tensor x) {
        return impl_->forward(x);
    }
};

// The implementation struct
struct BatchChannelNormalizationImpl : torch::nn::Module {
    int64_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;

    // Learnable parameters (if affine is true)
    torch::Tensor gamma_; // scale, named "weight" in PyTorch's BatchNorm
    torch::Tensor beta_;  // shift, named "bias" in PyTorch's BatchNorm

    // Buffers for running statistics (if track_running_stats is true)
    torch::Tensor running_mean_;
    torch::Tensor running_var_;
    torch::Tensor num_batches_tracked_; // Not strictly needed for forward, but PyTorch BN has it.

    BatchChannelNormalizationImpl(int64_t num_features, double eps = 1e-5, double momentum = 0.1,
                                  bool affine = true, bool track_running_stats = true)
        : num_features_(num_features),
          eps_(eps),
          momentum_(momentum),
          affine_(affine),
          track_running_stats_(track_running_stats) {
        TORCH_CHECK(num_features > 0, "num_features must be positive.");

        if (affine_) {
            gamma_ = register_parameter("weight", torch::ones({num_features_}));
            beta_ = register_parameter("bias", torch::zeros({num_features_}));
        } else {
            // Still register them as undefined tensors if not affine, PyTorch modules often do this.
            // Or simply don't declare them. For simplicity here, they won't exist if not affine.
        }

        if (track_running_stats_) {
            running_mean_ = register_buffer("running_mean", torch::zeros({num_features_}));
            running_var_ = register_buffer("running_var", torch::ones({num_features_})); // Start with variance 1
            num_batches_tracked_ = register_buffer("num_batches_tracked", torch::tensor(0, torch::kLong));
        } else {
            // Buffers won't exist if not tracking.
        }
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input x: (N, C, D1, D2, ...) where C is num_features_
        // N: Batch size
        // C: Number of features/channels
        // D1, D2, ...: Spatial or sequential dimensions

        TORCH_CHECK(x.dim() >= 2, "Input tensor must have at least 2 dimensions (N, C, ...). Got ", x.dim());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Number of input features (channels) mismatch. Expected ", num_features_,
                    ", but got ", x.size(1), " for input of shape ", x.sizes());

        torch::Tensor current_mean;
        torch::Tensor current_var;

        // Determine dimensions for calculating mean/variance:
        // Exclude channel dimension (dim 1). Include batch (dim 0) and all spatial/sequential dims (2, 3, ...).
        std::vector<int64_t> reduce_dims_for_stats;
        reduce_dims_for_stats.push_back(0); // Batch dimension
        for (int64_t i = 2; i < x.dim(); ++i) {
            reduce_dims_for_stats.push_back(i);
        }

        if (this->is_training() && track_running_stats_) {
            // Calculate mean and variance over the current batch
            // The result of mean/var over these dims should be of shape (num_features_)
            torch::Tensor batch_mean = x.mean(reduce_dims_for_stats, /*keepdim=false*/ false);
            // For variance, PyTorch's functional batch_norm uses N rather than N-1 for batch variance.
            // x.var(unbiased=false) would be sum((x-mean)^2)/numel_per_channel
            // Let N_total be the number of elements over which we average for each channel (N * D1 * D2 * ...)
            int64_t N_total_per_channel = 1;
            for(int64_t d : reduce_dims_for_stats) N_total_per_channel *= x.size(d);
            if (N_total_per_channel == 1 && x.dim() > 2) { // Only one spatial element per batch entry
                 // var will be 0, which is fine with eps.
                 // This matches PyTorch behavior (e.g., BatchNorm2d on 1x1 image)
            }
            // For batch_var, we need E[X^2] - (E[X])^2
            // torch::Tensor batch_var = x.var(reduce_dims_for_stats, /*unbiased=*/false, /*keepdim=false*/ false);
            // Safer way:
            auto x_minus_mean_sq = (x - batch_mean.view({1, num_features_, 1, 1})).pow(2); // Reshape for broadcast
            torch::Tensor batch_var = x_minus_mean_sq.mean(reduce_dims_for_stats, /*keepdim=false*/ false);


            // Update running statistics
            // The momentum in PyTorch's BatchNorm is for the running_mean and running_var,
            // not a simple average.
            // running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            // running_var  = (1 - momentum) * running_var  + momentum * batch_var
            // No .data() needed for buffers if not in no_grad block
            running_mean_ = (1.0 - momentum_) * running_mean_ + momentum_ * batch_mean;
            running_var_  = (1.0 - momentum_) * running_var_  + momentum_ * batch_var;

            if (num_batches_tracked_) { // Check if defined
                num_batches_tracked_ += 1;
            }

            current_mean = batch_mean;
            current_var = batch_var;
        } else {
            if (track_running_stats_) {
                current_mean = running_mean_;
                current_var = running_var_;
            } else {
                // No tracking, compute stats on current batch but don't update running stats
                // This mode is less common for BN but possible if track_running_stats=false
                current_mean = x.mean(reduce_dims_for_stats, /*keepdim=false*/ false);
                // current_var  = x.var(reduce_dims_for_stats, /*unbiased=*/false, /*keepdim=false*/ false);
                auto x_minus_mean_sq = (x - current_mean.view({1, num_features_, 1, 1})).pow(2);
                current_var = x_minus_mean_sq.mean(reduce_dims_for_stats, /*keepdim=false*/ false);
            }
        }

        // Reshape mean and var to (1, C, 1, 1, ...) for broadcasting
        std::vector<int64_t> view_shape(x.dim(), 1);
        if (x.dim() > 1) { // Should always be true
            view_shape[1] = num_features_;
        } else { // Should not happen given dim check
            view_shape[0] = num_features_;
        }

        torch::Tensor x_normalized = (x - current_mean.view(view_shape)) / torch::sqrt(current_var.view(view_shape) + eps_);

        if (affine_) {
            return x_normalized * gamma_.view(view_shape) + beta_.view(view_shape);
        } else {
            return x_normalized;
        }
    }

    // Optional: for pretty printing the module
    void pretty_print(std::ostream& stream) const override {
        stream << "BatchChannelNormalization(num_features=" << num_features_
               << ", eps=" << eps_ << ", momentum=" << momentum_
               << ", affine=" << (affine_ ? "true" : "false")
               << ", track_running_stats=" << (track_running_stats_ ? "true" : "false") << ")";
    }
};
TORCH_MODULE(BatchChannelNormalization);


// --- Example Usage ---
int main() {
    torch::manual_seed(0);

    int64_t num_features = 3;
    BatchChannelNormalization bcn_module(num_features, /*eps=*/1e-5, /*momentum=*/0.1);
    // std::cout << bcn_module << std::endl;

    // --- Test Case 1: 4D input (like CNN features NCHW) ---
    std::cout << "--- Test Case 1: 4D input (NCHW) ---" << std::endl;
    int64_t N1 = 4, H1 = 5, W1 = 5;
    torch::Tensor input1 = torch::randn({N1, num_features, H1, W1}) * 2 + 3; // Give it some mean and std

    std::cout << "Initial running_mean: " << bcn_module->running_mean_ << std::endl;
    std::cout << "Initial running_var: " << bcn_module->running_var_ << std::endl;

    // Training pass
    bcn_module->train();
    torch::Tensor output1_train = bcn_module->forward(input1);
    std::cout << "Output1_train shape: " << output1_train.sizes() << std::endl;
    std::cout << "Output1_train [:,0,:,:] mean (should be ~0): " << output1_train.select(1,0).mean().item<double>() << std::endl;
    std::cout << "Output1_train [:,0,:,:] std (should be ~1): " << output1_train.select(1,0).std(false).item<double>() << std::endl;

    std::cout << "Updated running_mean: " << bcn_module->running_mean_ << std::endl;
    std::cout << "Updated running_var: " << bcn_module->running_var_ << std::endl;
    std::cout << "Num batches tracked: " << bcn_module->num_batches_tracked_ << std::endl;

    // Another training pass to see running stats update further
    torch::Tensor input1_next = torch::randn({N1, num_features, H1, W1}) * 1.5 + 1;
    bcn_module->forward(input1_next); // Discard output, just update stats
    std::cout << "Further updated running_mean: " << bcn_module->running_mean_ << std::endl;
    std::cout << "Further updated running_var: " << bcn_module->running_var_ << std::endl;
    std::cout << "Num batches tracked: " << bcn_module->num_batches_tracked_ << std::endl;


    // Evaluation pass (should use running_mean and running_var)
    bcn_module->eval();
    torch::Tensor output1_eval = bcn_module->forward(input1); // Use original input1
    std::cout << "Output1_eval shape: " << output1_eval.sizes() << std::endl;
    std::cout << "Output1_eval [:,0,:,:] mean (uses running stats): " << output1_eval.select(1,0).mean().item<double>() << std::endl;
    std::cout << "Output1_eval [:,0,:,:] std (uses running stats): " << output1_eval.select(1,0).std(false).item<double>() << std::endl;

    TORCH_CHECK(!torch::allclose(output1_train.select(1,0).mean(), output1_eval.select(1,0).mean()),
                "Train and Eval mode outputs means should differ for first channel.");

    // --- Test Case 2: 2D input (NC - like MLP features after Linear) ---
    std::cout << "\n--- Test Case 2: 2D input (NC) ---" << std::endl;
    BatchChannelNormalization bcn_module_2d(num_features);
    int64_t N2 = 10;
    torch::Tensor input2 = torch::randn({N2, num_features}) * 3 - 2;

    bcn_module_2d->train();
    torch::Tensor output2_train = bcn_module_2d->forward(input2);
    std::cout << "Output2_train shape: " << output2_train.sizes() << std::endl;
    std::cout << "Output2_train [:,1] mean (should be ~0): " << output2_train.select(1,1).mean().item<double>() << std::endl;
    std::cout << "Output2_train [:,1] std (should be ~1): " << output2_train.select(1,1).std(false).item<double>() << std::endl;
    std::cout << "Updated running_mean (2D): " << bcn_module_2d->running_mean_ << std::endl;
    std::cout << "Updated running_var (2D): " << bcn_module_2d->running_var_ << std::endl;

    bcn_module_2d->eval();
    torch::Tensor output2_eval = bcn_module_2d->forward(input2);
    std::cout << "Output2_eval [:,1] mean (uses running stats): " << output2_eval.select(1,1).mean().item<double>() << std::endl;
    std::cout << "Output2_eval [:,1] std (uses running stats): " << output2_eval.select(1,1).std(false).item<double>() << std::endl;


    // --- Test Case 3: No affine, no tracking ---
    std::cout << "\n--- Test Case 3: No affine, no tracking_stats ---" << std::endl;
    BatchChannelNormalization bcn_no_affine_no_track(num_features, 1e-5, 0.1, false, false);
    // std::cout << bcn_no_affine_no_track << std::endl;
    TORCH_CHECK(!bcn_no_affine_no_track->gamma_.defined(), "Gamma should not be defined.");
    TORCH_CHECK(!bcn_no_affine_no_track->running_mean_.defined(), "Running mean should not be defined.");

    bcn_no_affine_no_track->train(); // Mode doesn't matter much if not tracking
    torch::Tensor output3_train = bcn_no_affine_no_track->forward(input1);
    std::cout << "Output3_train [:,0,:,:] mean (batch stat, no affine, ~0): " << output3_train.select(1,0).mean().item<double>() << std::endl;
    std::cout << "Output3_train [:,0,:,:] std (batch stat, no affine, ~1): " << output3_train.select(1,0).std(false).item<double>() << std::endl;

    bcn_no_affine_no_track->eval(); // Should still use batch stats as track_running_stats=false
    torch::Tensor output3_eval = bcn_no_affine_no_track->forward(input1);
    TORCH_CHECK(torch::allclose(output3_train, output3_eval),
                "Outputs should be same if not tracking running stats, regardless of train/eval mode.");

    std::cout << "\nBatchChannelNormalization tests finished." << std::endl;
    return 0;
}


namespace xt::norm
{
    auto BatchChannelNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
