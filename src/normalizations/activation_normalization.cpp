#include "include/normalizations/activation_normalization.h"



#include <torch/torch.h>
#include <iostream>
#include <vector>

// Forward declaration for the Impl struct
struct ActivationNormalizationImpl;

// The main module struct that users will interact with.
// It uses TORCH_MODULE to handle the shared_ptr mechanics.
struct ActivationNormalization : torch::nn::ModuleHolder<ActivationNormalizationImpl> {
    using torch::nn::ModuleHolder<ActivationNormalizationImpl>::ModuleHolder;

    // The forward method is exposed through the Impl class
    torch::Tensor forward(torch::Tensor x) {
        return impl_->forward(x);
    }
};

// The implementation struct
struct ActivationNormalizationImpl : torch::nn::Module {
    // Parameters
    int64_t num_features_;
    double eps_;
    bool affine_;

    // Learnable parameters (if affine is true)
    torch::Tensor gamma_; // scale
    torch::Tensor beta_;  // shift

    ActivationNormalizationImpl(int64_t num_features, double eps = 1e-5, bool affine = true)
        : num_features_(num_features), eps_(eps), affine_(affine) {
        // num_features typically corresponds to the number of channels (C)
        // in an input of shape (N, C, H, W) or (N, C, L)

        if (affine_) {
            // In PyTorch, Batch/Instance/LayerNorm often call scale 'weight' and shift 'bias'
            gamma_ = register_parameter("weight", torch::ones({num_features_}));
            beta_  = register_parameter("bias",   torch::zeros({num_features_}));
        }
        // No running_mean or running_var needed for Instance Normalization
        // as statistics are computed per instance.
    }

    torch::Tensor forward(torch::Tensor x) {
        // Input x is expected to be at least 2D: (N, C, ...)
        // N: Batch size
        // C: Number of features/channels (must match num_features_)
        // ...: Spatial or sequential dimensions (e.g., H, W or L)

        TORCH_CHECK(x.dim() >= 2, "Input tensor must have at least 2 dimensions (Batch, Channels, ...). Got ", x.dim());
        TORCH_CHECK(x.size(1) == num_features_,
                    "Number of input features (channels) mismatch. Expected ", num_features_,
                    ", but got ", x.size(1), " for input of shape ", x.sizes());

        // We want to normalize over the spatial/sequential dimensions for each instance and each channel.
        // Dimensions to reduce over start from dimension 2.
        std::vector<int64_t> reduce_dims;
        for (int64_t i = 2; i < x.dim(); ++i) {
            reduce_dims.push_back(i);
        }

        // If there are no spatial/sequential dimensions (e.g. input is (N, C)),
        // then mean will be the value itself and var will be 0.
        // This is typically handled okay by the epsilon, but it's worth noting.
        // For an input (N, C), reduce_dims will be empty.
        // PyTorch's mean/var over empty dims return the tensor itself, which isn't what we want for instance norm.
        // InstanceNorm for (N,C) is ill-defined unless C=1, where it becomes a no-op before affine.
        // Let's assume there are spatial/sequential dimensions (dim >= 3) or it's (N, C, 1, 1...)
        if (x.dim() > 2 && reduce_dims.empty()) { // Should not happen if x.dim() > 2
             // This case is unlikely given the loop structure.
        } else if (x.dim() == 2 && reduce_dims.empty()) {
            // For (N,C) input:
            // InstanceNorm is often not applied or would mean normalizing over a single point if C=num_features.
            // PyTorch's InstanceNorm1d expects (N, C, L).
            // If we truly want to normalize (N,C) as if L=1:
            // mean will be x itself, var will be 0.
            // (x - x) / sqrt(0 + eps) = 0. Then scaled by gamma and shifted by beta.
            // This is how torch.nn.InstanceNorm1d behaves if L=1.
            // We can simulate this by adding and removing a dummy dimension.
            bool was_2d = false;
            if (x.dim() == 2) {
                x = x.unsqueeze(-1); // (N, C) -> (N, C, 1)
                reduce_dims.push_back(2); // new dimension to reduce
                was_2d = true;
            }


            // Calculate mean and variance per instance, per channel.
            // keepdim=true ensures that the output tensors have the same number of dimensions
            // as the input, which simplifies broadcasting.
            auto mean = x.mean(reduce_dims, /*keepdim=*/true);
            // unbiased=false for population variance, usually preferred for normalization layers.
            auto var = x.var(reduce_dims, /*unbiased=*/false, /*keepdim=*/true);

            auto x_normalized = (x - mean) / torch::sqrt(var + eps_);

            if (was_2d) {
                x_normalized = x_normalized.squeeze(-1); // (N, C, 1) -> (N, C)
                // x is already squeezed implicitly by next ops if affine
            }


            if (affine_) {
                // gamma_ and beta_ are of shape (num_features_).
                // They need to be reshaped to (1, num_features_, 1, 1, ...) to broadcast correctly
                // with x_normalized, which has shape (N, num_features_, D1, D2, ...).

                std::vector<int64_t> affine_param_shape(x_normalized.dim(), 1); // e.g., {1, 1} for 2D, {1,1,1} for 3D
                if (x_normalized.dim() > 1) { // Should always be true
                    affine_param_shape[1] = num_features_; // e.g., {1, C} or {1, C, 1}
                } else { // Special case for (C) input if x_normalized became 1D
                    affine_param_shape[0] = num_features_;
                }


                return x_normalized * gamma_.view(affine_param_shape) + beta_.view(affine_param_shape);
            } else {
                return x_normalized;
            }
        }
        // Fallback for cases not perfectly handled, or if logic above is too complex,
        // use torch::instance_norm directly if it simplifies.
        // However, the goal is to implement it manually. The above logic should cover it.
        // The previous simple path was fine for dim > 2. The 2D case complexity is to match PyTorch InstanceNorm1d(L=1).

        // --- Simplified version for input.dim() > 2 ---
        if (x.dim() > 2) { // Original path for N, C, D1, ...
            auto mean = x.mean(reduce_dims, /*keepdim=*/true);
            auto var = x.var(reduce_dims, /*unbiased=*/false, /*keepdim=*/true);
            auto x_normalized = (x - mean) / torch::sqrt(var + eps_);

            if (affine_) {
                std::vector<int64_t> affine_param_shape(x.dim(), 1);
                affine_param_shape[1] = num_features_;
                return x_normalized * gamma_.view(affine_param_shape) + beta_.view(affine_param_shape);
            } else {
                return x_normalized;
            }
        } else { // x.dim() == 2, input is (N, C)
            // Replicate InstanceNorm1d behavior for L=1
            // (x - x) / sqrt(0 + eps) = 0
            // Then apply affine transform: 0 * gamma + beta = beta
            // This is quite specific. A simple channel-wise Z-score over batch might be more useful
            // but that's BatchNorm not InstanceNorm.
            // For true InstanceNorm-like behavior on (N,C) where C is features:
            // it means each feature vector (row) is normalized. This is more like LayerNorm on features.
            // Given the typical use of InstanceNorm (N, C, Spatial), the (N,C) case
            // means each of the C channels for a given N has only one value.
            // Normalizing a single value: (v - v) / sqrt(0 + eps) = 0.
            auto x_normalized = torch::zeros_like(x);
             if (affine_) {
                std::vector<int64_t> affine_param_shape = {1, num_features_}; // For (N,C) input
                return x_normalized * gamma_.view(affine_param_shape) + beta_.view(affine_param_shape);
             } else {
                return x_normalized;
             }
        }
    }

    // Optional: for pretty printing the module
    void pretty_print(std::ostream& stream) const override {
        stream << "ActivationNormalization(num_features=" << num_features_
               << ", eps=" << eps_ << ", affine=" << (affine_ ? "true" : "false") << ")";
    }
};
TORCH_MODULE(ActivationNormalization); // Creates the ActivationNormalization wrapper from ActivationNormalizationImpl


// --- Example Usage ---
int main() {
    // --- Test Case 1: 4D input (like CNN) ---
    std::cout << "--- Test Case 1: 4D input (image-like) ---" << std::endl;
    int64_t num_features_4d = 3; // e.g., 3 color channels
    int64_t batch_size_4d = 2;
    int64_t height_4d = 4;
    int64_t width_4d = 4;

    ActivationNormalization act_norm_4d(num_features_4d);
    // std::cout << act_norm_4d << std::endl; // For full parameter print

    // Create a dummy input tensor (Batch, Channels, Height, Width)
    torch::Tensor input_4d = torch::randn({batch_size_4d, num_features_4d, height_4d, width_4d});
    std::cout << "Input 4D shape: " << input_4d.sizes() << std::endl;

    torch::Tensor output_4d = act_norm_4d->forward(input_4d);
    std::cout << "Output 4D shape: " << output_4d.sizes() << std::endl;

    // Check statistics for the first instance, first channel (after normalization, before affine)
    // To do this properly, we'd need to re-calculate without affine or access internal x_normalized
    // For now, let's just observe one channel of the output.
    // If affine=true, mean won't be 0 and std won't be 1 unless gamma=1, beta=0.
    // Since gamma=1 and beta=0 by default, it should be normalized.
    std::cout << "Output 4D [0,0,:,:] mean: " << output_4d.select(0,0).select(0,0).mean().item<double>() << std::endl;
    std::cout << "Output 4D [0,0,:,:] std: " << output_4d.select(0,0).select(0,0).std().item<double>() << std::endl;
    std::cout << "Output 4D [0,1,:,:] mean: " << output_4d.select(0,0).select(0,1).mean().item<double>() << std::endl;
    std::cout << "Output 4D [0,1,:,:] std: " << output_4d.select(0,0).select(0,1).std().item<double>() << std::endl;


    // --- Test Case 2: 3D input (like 1D CNN or sequences) ---
    std::cout << "\n--- Test Case 2: 3D input (sequence-like) ---" << std::endl;
    int64_t num_features_3d = 64; // e.g., embedding dimension as channels
    int64_t batch_size_3d = 4;
    int64_t seq_len_3d = 10;

    ActivationNormalization act_norm_3d(num_features_3d, 1e-5, /*affine=*/true);
    torch::Tensor input_3d = torch::randn({batch_size_3d, num_features_3d, seq_len_3d});
    std::cout << "Input 3D shape: " << input_3d.sizes() << std::endl;

    torch::Tensor output_3d = act_norm_3d->forward(input_3d);
    std::cout << "Output 3D shape: " << output_3d.sizes() << std::endl;
    std::cout << "Output 3D [0,0,:] mean: " << output_3d.select(0,0).select(0,0).mean().item<double>() << std::endl;
    std::cout << "Output 3D [0,0,:] std: " << output_3d.select(0,0).select(0,0).std().item<double>() << std::endl;

    // --- Test Case 3: 2D input (Batch, Features) ---
    // Note: InstanceNorm on (Batch, Features) is a bit unusual.
    // PyTorch's InstanceNorm1d expects (N, C, L). If L=1, then it normalizes a single point per channel.
    // (value - value) / sqrt(0 + eps) = 0. Then affine transform.
    std::cout << "\n--- Test Case 3: 2D input (N, C) ---" << std::endl;
    int64_t num_features_2d = 5;
    int64_t batch_size_2d = 3;

    ActivationNormalization act_norm_2d(num_features_2d);
    torch::Tensor input_2d = torch::randn({batch_size_2d, num_features_2d});
    std::cout << "Input 2D shape: " << input_2d.sizes() << std::endl;
    // Set gamma and beta to something non-default for 2D to see effect
    act_norm_2d->gamma_.data().fill_(2.0);
    act_norm_2d->beta_.data().fill_(0.5);


    torch::Tensor output_2d = act_norm_2d->forward(input_2d);
    std::cout << "Output 2D shape: " << output_2d.sizes() << std::endl;
    std::cout << "Output 2D (should be all beta values if logic matches InstanceNorm1d(L=1)): \n" << output_2d << std::endl;
    // Expected output for InstanceNorm1d with L=1 input: beta
    // Our logic for 2D should make x_normalized = 0, so output = gamma * 0 + beta = beta.

    // --- Test Case 4: No affine parameters ---
    std::cout << "\n--- Test Case 4: No affine parameters ---" << std::endl;
    ActivationNormalization act_norm_no_affine(num_features_4d, 1e-5, /*affine=*/false);
    torch::Tensor output_no_affine = act_norm_no_affine->forward(input_4d);
    std::cout << "Output 4D (no affine) [0,0,:,:] mean: " << output_no_affine.select(0,0).select(0,0).mean().item<double>() << std::endl;
    std::cout << "Output 4D (no affine) [0,0,:,:] std: " << output_no_affine.select(0,0).select(0,0).std(false).item<double>() << std::endl; // std(false) for population std

    return 0;
}



namespace xt::norm
{
    auto ActiveNorm::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return torch::zeros(10);
    }
}
