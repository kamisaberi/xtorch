#include "include/dropouts/drop_block.h"



#include <torch/torch.h>
#include <vector>
#include <algorithm> // For std::min, std::max
#include <cmath>     // For std::pow, std::floor
#include <ostream>   // For std::ostream

struct DropBlockImpl : torch::nn::Module {
    int block_size_;
    double drop_prob_; // This is the target probability 'p' for dropping units, similar to standard Dropout
    double epsilon_ = 1e-6; // For numerical stability in division

    DropBlockImpl(int block_size = 7, double drop_prob = 0.1)
        : block_size_(block_size), drop_prob_(drop_prob) {
        TORCH_CHECK(block_size_ > 0, "DropBlock block_size must be positive.");
        TORCH_CHECK(drop_prob_ >= 0.0 && drop_prob_ <= 1.0, "DropBlock drop_prob must be between 0 and 1.");
    }

    torch::Tensor forward(const torch::Tensor& input) {
        if (!this->is_training() || drop_prob_ == 0.0 || block_size_ == 0) {
            return input;
        }
        if (drop_prob_ == 1.0) {
            return torch::zeros_like(input);
        }

        auto input_sizes = input.sizes();
        TORCH_CHECK(input_sizes.size() == 4, "DropBlock expects a 4D input tensor (N, C, H, W).");
        int64_t N = input_sizes[0];
        int64_t C = input_sizes[1];
        int64_t H = input_sizes[2];
        int64_t W = input_sizes[3];

        // If block_size is larger than or equal to feature map size,
        // it acts like standard spatial dropout (dropping entire feature maps).
        // For simplicity, we can clamp block_size, or handle gamma calculation carefully.
        // The paper's gamma calculation handles this.
        if (block_size_ > H || block_size_ > W) {
            // Fallback to standard dropout behavior if block is too large.
            // This is a simplification; a more precise DropBlock would still operate.
            // Or, one could simply not apply dropout if block_size is too large relative to one dimension.
            // For now, let's proceed and gamma calculation should handle it if seed_area becomes 0 or small.
            // A common strategy is to cap block_size at min(H,W) if it exceeds.
            // However, the gamma formula naturally handles cases where block_size is large.
        }

        // Calculate gamma: the probability of a seed point in the downsampled grid being 1 (start of a dropped block).
        // gamma = (drop_prob / block_size^2) * (feat_map_area / valid_seed_region_area)
        double feat_map_area = static_cast<double>(H * W);
        double block_area = static_cast<double>(block_size_ * block_size_);

        // Seed region dimensions
        // Clamping seed_H/W to be at least 1 if block_size is very large.
        // (H - block_size_ + 1) can be <=0 if block_size_ > H.
        double seed_H_double = std::max(1.0, static_cast<double>(H - block_size_ + 1));
        double seed_W_double = std::max(1.0, static_cast<double>(W - block_size_ + 1));
        double valid_seed_region_area = seed_H_double * seed_W_double;

        // Ensure block_area and valid_seed_region_area are not zero to avoid division by zero.
        if (block_area < epsilon_ || valid_seed_region_area < epsilon_) {
             // This case implies block_size is 0 or feature map area for seeds is 0.
             // Effectively, can't apply DropBlock meaningfully.
             // Fallback to simpler dropout (e.g. drop all if drop_prob > 0.5, else keep all) or no dropout.
             if (drop_prob_ > 0.0) { // If trying to drop, and block mechanics fail, just drop with p.
                 // This is a rough approximation of SpatialDropout.
                 torch::Tensor p_tensor = torch::full({N, C, 1, 1}, drop_prob_, input.options());
                 torch::Tensor bernoulli_mask = torch::bernoulli(1.0 - p_tensor); // keep prob
                 return input * bernoulli_mask / (1.0 - drop_prob_ + epsilon_);
             }
             return input; // No dropout
        }

        double gamma = (drop_prob_ / block_area) * (feat_map_area / valid_seed_region_area);
        gamma = std::min(1.0, std::max(0.0, gamma)); // Clamp gamma to [0,1]

        int64_t seed_H = static_cast<int64_t>(std::floor(seed_H_double));
        int64_t seed_W = static_cast<int64_t>(std::floor(seed_W_double));

        // Create the mask for block centers (seed mask)
        // Shape (N, C, seed_H, seed_W)
        torch::Tensor mask_seeds = torch::bernoulli(torch::full({N, C, seed_H, seed_W}, gamma, input.options().dtype(torch::kFloat32)));

        // Expand the seed mask to the full block mask using max_pooling
        // The padding centers the block around the seed.
        int padding = block_size_ / 2; // Integer division
        torch::Tensor block_mask_small = torch::nn::functional::max_pool2d(
            mask_seeds,
            torch::nn::functional::MaxPool2dFuncOptions(block_size_).stride(1).padding(padding)
        );
        // block_mask_small is now of shape (N, C, seed_H, seed_W), where each 1 represents a dropped block region
        // if its center was a seed.

        // Pad block_mask_small to match input's H, W dimensions, effectively centering the DropBlock active region.
        int64_t delta_H = H - seed_H;
        int64_t pad_top = delta_H / 2;
        int64_t pad_bottom = delta_H - pad_top;

        int64_t delta_W = W - seed_W;
        int64_t pad_left = delta_W / 2;
        int64_t pad_right = delta_W - pad_left;

        torch::Tensor drop_pattern_mask; // This will be 1 for dropped, 0 for kept.
        if (pad_top < 0 || pad_bottom < 0 || pad_left < 0 || pad_right < 0) {
            // This happens if seed_H > H (i.e. block_size_ < 1 or similar degenerate cases for seed dim calc).
            // Should be caught by earlier checks or implies block_size_ is very small relative to H, W.
            // If block_mask_small is already larger than H,W due to pooling with large block_size on small seed_grid,
            // we need to crop it.
            // For now, assume seed_H <= H.
            // A robust version would use F::interpolate or crop if block_mask_small dimensions mismatch.
            // Given the definition of seed_H/W, they should be <= H/W.
             drop_pattern_mask = torch::nn::functional::pad(
                block_mask_small,
                torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom}).mode(torch::kConstant).value(0)
            );
        } else {
             drop_pattern_mask = torch::nn::functional::pad(
                block_mask_small,
                torch::nn::functional::PadFuncOptions({pad_left, pad_right, pad_top, pad_bottom}).mode(torch::kConstant).value(0)
            );
        }

        // Ensure the padded mask has the correct final dimensions H, W.
        // It's possible due to padding logic and integer arithmetic that it's off by one.
        // A crop or careful F::interpolate would be more robust here if exact H,W are critical from padding.
        // For simplicity, we assume padding gets it to H,W. If not, crop:
        if (drop_pattern_mask.size(2) != H || drop_pattern_mask.size(3) != W) {
             drop_pattern_mask = drop_pattern_mask.index({
                torch::indexing::Slice(), torch::indexing::Slice(),
                torch::indexing::Slice(0, H), torch::indexing::Slice(0, W)
            });
        }


        torch::Tensor keep_mask = 1.0 - drop_pattern_mask; // 1 for kept, 0 for dropped

        // Normalize the output: scale by (num_elements_in_mask_area / num_kept_elements_in_mask_area)
        // We normalize per feature map (N, C)
        torch::Tensor num_kept_per_map = keep_mask.sum({2, 3}, /*keepdim=*/true); // Sum over H, W. Result (N, C, 1, 1)
        torch::Tensor scale = static_cast<double>(H * W) / (num_kept_per_map + epsilon_);

        return input * keep_mask * scale;
    }

    void pretty_print(std::ostream& stream) const override {
        stream << "DropBlock(block_size=" << block_size_
               << ", drop_prob=" << drop_prob_ << ")";
    }
};

TORCH_MODULE(DropBlock); // Creates the DropBlock module "class"

/*
// Example of how to use the DropBlock module:
// (This is for illustration and would typically be in your main application code)

#include <iostream>

void run_drop_block_example() {
    torch::manual_seed(0); // For reproducible results

    int block_s = 3;
    double drop_p = 0.2;
    DropBlock drop_block_module(block_s, drop_p);
    std::cout << "DropBlock Module: " << drop_block_module << std::endl;

    // Example input tensor (Batch, Channels, Height, Width)
    torch::Tensor input_tensor = torch::ones({1, 1, 10, 10}); // Single 10x10 feature map
    std::cout << "Input Tensor (1x1x10x10 of ones)" << std::endl;

    // --- Training mode ---
    drop_block_module->train(); // Set the module to training mode
    torch::Tensor output_train = drop_block_module->forward(input_tensor);
    std::cout << "Output (training mode):\n" << output_train.squeeze() << std::endl;
    // Expected: Some 3x3 blocks will be zeroed out. Non-zero elements will be scaled.

    double num_zeros = (output_train == 0).sum().item<double>();
    double total_elements = output_train.numel();
    std::cout << "Percentage of zeros: " << (num_zeros / total_elements) * 100.0 << "%" << std::endl;
    // Note: The actual percentage of zeros might not exactly match drop_prob * block_size^2 / area
    // due to overlaps and the stochastic nature of gamma. The normalization aims to keep expectation correct.

    // --- Evaluation mode ---
    drop_block_module->eval(); // Set the module to evaluation mode
    torch::Tensor output_eval = drop_block_module->forward(input_tensor);
    std::cout << "\nOutput (evaluation mode):\n" << output_eval.squeeze() << std::endl;
    // Expected output to be identical to input in evaluation mode.
    TORCH_CHECK(torch::allclose(input_tensor, output_eval), "DropBlock eval output mismatch!");


    // Example with larger drop probability
    DropBlock high_drop_module(block_s, 0.5);
    high_drop_module->train();
    torch::Tensor output_high_drop = high_drop_module->forward(input_tensor);
    std::cout << "\nOutput (training mode, high drop_prob=0.5):\n" << output_high_drop.squeeze() << std::endl;
    num_zeros = (output_high_drop == 0).sum().item<double>();
    std::cout << "Percentage of zeros (high drop_prob): " << (num_zeros / total_elements) * 100.0 << "%" << std::endl;

     // Example where block size is large
    DropBlock large_block_module(10, 0.2); // Block size is same as feature map size
    large_block_module->train();
    torch::Tensor output_large_block = large_block_module->forward(input_tensor);
    std::cout << "\nOutput (training mode, block_size=10, H=W=10):\n" << output_large_block.squeeze() << std::endl;
    // Expected: The entire 10x10 map will either be all zeros (scaled) or all ones (scaled),
    // depending on the single Bernoulli trial for its seed.
    // If it's all zeros, output sum should be 0. If kept, output sum should be scaled input sum.
    if (output_large_block.sum().item<float>() == 0) {
        std::cout << "Large block was dropped." << std::endl;
    } else {
        std::cout << "Large block was kept (and scaled)." << std::endl;
         // Check scale: sum(input*scale) = 1*1*10*10 * (10*10 / (10*10 - 0)) = 100
        TORCH_CHECK(std::abs(output_large_block.sum().item<float>() - input_tensor.sum().item<float>() * ( (10.0*10.0) / ( (10.0*10.0 - 0*0) + 1e-6) ) ) < 1.0 , "Large block scale mismatch");
    }
}

// int main() {
//    run_drop_block_example();
//    return 0;
// }
*/


namespace xt::dropouts
{
    torch::Tensor drop_block(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto DropBlock::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::drop_block(torch::zeros(10));
    }
}
