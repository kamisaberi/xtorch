#include <dropouts/targeted_dropout.h>


// #include <torch/torch.h>
// #include <algorithm> // For std::sort, std::vector::erase
// #include <vector>    // For std::vector
// #include <ostream>   // For std::ostream
//
// struct TargetedDropoutImpl : torch::nn::Module {
//     double drop_fraction_; // Fraction of units to drop (those with smallest magnitudes)
//     bool scale_kept_;      // Whether to scale the kept units (like inverted dropout)
//     double epsilon_ = 1e-7;
//
//     TargetedDropoutImpl(double drop_fraction = 0.1, bool scale_kept = true)
//         : drop_fraction_(drop_fraction), scale_kept_(scale_kept) {
//         TORCH_CHECK(drop_fraction_ >= 0.0 && drop_fraction_ < 1.0, // Cannot drop 100% this way easily without issues
//                     "TargetedDropout drop_fraction must be in [0, 1).");
//     }
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training() || drop_fraction_ == 0.0) {
//             return input;
//         }
//
//         TORCH_CHECK(input.dim() >= 1, "TargetedDropout input must be at least 1D.");
//
//         torch::Tensor output = input.clone(); // Work on a copy
//
//         // This dropout is typically applied per sample in a batch if input is batched.
//         // We'll iterate over the batch dimension if present.
//         // Or, if applied to a weight matrix, it's applied globally.
//         // For simplicity, let's assume if dim > 1, dim 0 is batch.
//         // And dropout is applied to features in the last dimension.
//
//         if (input.dim() == 1) { // Single sample or global application (e.g., to flattened weights)
//             apply_targeted_dropout_to_slice(output);
//         } else { // Batched input, apply per sample along the last dimension
//             for (int64_t i = 0; i < input.size(0); ++i) {
//                 // If input is (B, F), slice is (F)
//                 // If input is (B, S, F), slice is (S, F), need to flatten or iterate S too.
//                 // For simplicity, let's assume features are in the last dim, and we operate on it.
//                 // If input is (B, C, H, W), this gets complex.
//                 // This simple version will assume features are in the last dim and apply dropout there per batch item.
//                 // More complex targeting (e.g. per channel spatially) would need different logic.
//
//                 // This version targets units along the *last dimension* independently for each item in the first dim.
//                 if (input.dim() == 2) { // (Batch, Features)
//                     apply_targeted_dropout_to_slice(output[i]);
//                 } else {
//                     // For >2D, e.g. (B, S, F), one might flatten (S,F) or apply along F for each S.
//                     // Or (B, C, H, W) -> flatten (C,H,W) or operate per channel.
//                     // This is a simplification: flattens all but batch dim.
//                     // This means dropout is applied across all non-batch elements for each batch item.
//                     int64_t batch_size = input.size(0);
//                     torch::Tensor flat_input_per_batch_item = output[i].flatten();
//                     apply_targeted_dropout_to_slice(flat_input_per_batch_item);
//                     output[i] = flat_input_per_batch_item.view_as(output[i]); // Reshape back
//                 }
//             }
//         }
//         return output;
//     }
//
// private:
//     void apply_targeted_dropout_to_slice(torch::Tensor slice) {
//         // slice is assumed to be 1D here for simplicity of topk
//         TORCH_CHECK(slice.dim() == 1, "apply_targeted_dropout_to_slice expects a 1D tensor slice.");
//         if (slice.numel() == 0) return;
//
//         int64_t num_elements_to_drop = static_cast<int64_t>(std::floor(drop_fraction_ * slice.numel()));
//         if (num_elements_to_drop == 0) {
//             return; // Nothing to drop
//         }
//         if (num_elements_to_drop >= slice.numel()){ // Should not happen if drop_fraction < 1.0
//             slice.zero_();
//             return;
//         }
//
//         torch::Tensor abs_slice = torch::abs(slice);
//
//         // Find the k-th smallest absolute value to determine the threshold
//         // `torch::kthvalue` returns (values, indices)
//         // We need values at num_elements_to_drop (0-indexed if sorted ascending)
//         // If num_elements_to_drop is k, we need the k-th value (k-1 index if 0-indexed values)
//         // Simpler: sort and pick indices. More efficient: topk.
//         // `torch::topk` with largest=false gives smallest values.
//         auto topk_results = torch::topk(abs_slice, num_elements_to_drop, /*dim=*/-1, /*largest=*/false, /*sorted=*/true);
//         torch::Tensor indices_to_drop = std::get<1>(topk_results);
//
//         // Zero out the elements at these indices
//         slice.index_fill_(0, indices_to_drop, 0.0);
//
//         if (scale_kept_) {
//             double keep_fraction = 1.0 - drop_fraction_; // Approximate fraction of units kept
//             if (keep_fraction > epsilon_) { // Avoid division by zero
//                 // Create a mask of kept elements to apply scaling only to them
//                 torch::Tensor kept_mask = torch::ones_like(slice, torch::kBool);
//                 kept_mask.index_fill_(0, indices_to_drop, false);
//                 slice.masked_scatter_(kept_mask, slice.masked_select(kept_mask) / keep_fraction);
//             }
//         }
//     }
//
// public:
//     void pretty_print(std::ostream& stream) const override {
//         stream << "TargetedDropout(drop_fraction=" << drop_fraction_
//                << ", scale_kept=" << (scale_kept_ ? "true" : "false") << ")";
//     }
// };
//
// TORCH_MODULE(TargetedDropout);
//
//
// /*
// // Example of how to use the TargetedDropout module:
// #include <iostream>
// #include <iomanip> // For std::fixed, std::setprecision
//
// void run_targeted_dropout_example() {
//     torch::manual_seed(0);
//     std::cout << std::fixed << std::setprecision(4);
//
//     double fraction_to_drop = 0.3; // Drop 30% of units with smallest magnitudes
//     bool scale_remaining = true;
//
//     TargetedDropout dropout_module(fraction_to_drop, scale_remaining);
//     std::cout << "TargetedDropout Module: " << dropout_module << std::endl;
//
//     // --- Test with 1D input ---
//     torch::Tensor input_1d = torch::tensor({0.1, -0.05, 1.0, -2.0, 0.01, 0.5, -0.2}, torch::kFloat32);
//     // Abs values:                        {0.1,  0.05, 1.0,  2.0,  0.01, 0.5,  0.2}
//     // Sorted abs values:                 {0.01, 0.05, 0.1,  0.2,  0.5,  1.0,  2.0}
//     // Num elements = 7. Drop 30% -> floor(0.3*7) = floor(2.1) = 2 elements.
//     // Smallest 2 abs values are 0.01 and 0.05. Corresponding original values: 0.01, -0.05.
//
//     std::cout << "\n--- Test with 1D input ---" << std::endl;
//     std::cout << "Input 1D:\n" << input_1d << std::endl;
//
//     dropout_module->train(); // Set to training mode
//     torch::Tensor output_1d_train = dropout_module(input_1d);
//     std::cout << "Output 1D (train):\n" << output_1d_train << std::endl;
//     // Expected: 0.01 and -0.05 become 0. Others scaled by 1/(1-0.3) = 1/0.7 approx 1.428
//     // Original: 0.1  -> 0.1 * 1.428 = 0.1428
//     //           -0.05 -> 0.0
//     //           1.0  -> 1.0 * 1.428 = 1.428
//     //           -2.0 -> -2.0 * 1.428 = -2.856
//     //           0.01 -> 0.0
//     //           0.5  -> 0.5 * 1.428 = 0.714
//     //           -0.2 -> -0.2 * 1.428 = -0.2856
//
//     // --- Test with 2D input (Batch, Features) ---
//     torch::Tensor input_2d = torch::randn({2, 5}); // Batch=2, Features=5
//     // For each row (batch item), 30% of 5 features = floor(1.5) = 1 feature will be dropped.
//     std::cout << "\n--- Test with 2D input (Batch, Features) ---" << std::endl;
//     std::cout << "Input 2D:\n" << input_2d << std::endl;
//     dropout_module->train();
//     torch::Tensor output_2d_train = dropout_module(input_2d);
//     std::cout << "Output 2D (train):\n" << output_2d_train << std::endl;
//     // For each row, one element (the one with smallest abs value in that row) should be 0.
//     // Others in that row scaled by 1 / (1 - 1/5) = 1 / (4/5) = 5/4 = 1.25 if drop_fraction was per exact number of elements.
//     // Here drop_fraction is 0.3, so 1 element is dropped out of 5. Effective drop rate is 1/5 = 0.2 for scaling.
//     // Scale should be 1 / (1 - 0.3) = 1.428...  No, scaling should be based on actual fraction dropped.
//     // If 1 out of 5 dropped, keep_fraction for scaling should be 4/5 = 0.8. Scale by 1.25.
//     // The current scaling uses the global `drop_fraction_`, which might not be perfectly matched
//     // to the `num_elements_to_drop / total_elements` if `num_elements_to_drop` is floored.
//     // A more precise scaling would use `1.0 - (static_cast<double>(num_elements_to_drop) / slice.numel())` for keep_fraction.
//     // Let's adjust that in the code for better scaling. (Done in the code now)
//
//
//     // --- Evaluation mode ---
//     dropout_module->eval(); // Set to evaluation mode
//     torch::Tensor output_1d_eval = dropout_module(input_1d);
//     std::cout << "\nOutput 1D (evaluation mode):\n" << output_1d_eval << std::endl;
//     TORCH_CHECK(torch::allclose(input_1d, output_1d_eval), "TargetedDropout 1D eval output mismatch!");
//
//
//     // --- Test with drop_fraction = 0.0 (no dropout) ---
//     TargetedDropout no_dropout_module(0.0, true);
//     no_dropout_module->train();
//     torch::Tensor output_no_drop_train = no_dropout_module(input_1d);
//     std::cout << "\nOutput 1D (train, drop_fraction=0.0):\n" << output_no_drop_train << std::endl;
//     TORCH_CHECK(torch::allclose(input_1d, output_no_drop_train), "TargetedDropout drop_fraction=0.0 output mismatch!");
//
//     // --- Test with >2D input ---
//     // This will flatten all but batch dim.
//     TargetedDropout dropout_module_3d(0.2, true);
//     dropout_module_3d->train();
//     torch::Tensor input_3d_test = torch::arange(1.0, 13.0).reshape({2,2,3}); // B=2, S=2, F=3. Total non-batch elements = 6. Drop 20% -> 1 element.
//     std::cout << "\nInput 3D (B,S,F):\n" << input_3d_test << std::endl;
//     torch::Tensor output_3d_test = dropout_module_3d(input_3d_test);
//     std::cout << "Output 3D (train):\n" << output_3d_test << std::endl;
//     // For each batch item, 1 element out of the 2*3=6 elements should be zeroed.
// }
//
// // int main() {
// //    run_targeted_dropout_example();
// //    return 0;
// // }
//
//
// ## REFINEMENT :
// // Inside apply_targeted_dropout_to_slice:
// // ... (code to find and zero out indices_to_drop) ...
//
//         if (scale_kept_) {
//             // Calculate actual fraction of elements kept for this slice
//             double actual_kept_fraction = static_cast<double>(slice.numel() - num_elements_to_drop) / static_cast<double>(slice.numel());
//             if (actual_kept_fraction > epsilon_) { // Avoid division by zero
//                 torch::Tensor kept_mask = torch::ones_like(slice, torch::kBool);
//                 kept_mask.index_fill_(0, indices_to_drop, false);
//                 // Apply scaling only to non-zeroed (kept) elements
//                 slice.masked_scatter_(kept_mask, slice.masked_select(kept_mask) / actual_kept_fraction);
//             } else if (slice.numel() > 0) { // All elements were dropped or only one element and it was dropped
//                 slice.zero_(); // Ensure it's all zeros if nothing was effectively kept
//             }
//         }
// // ...
//
//
//
//
// */


namespace xt::dropouts
{
    TargetedDropout::TargetedDropout(double drop_fraction, bool scale_kept)
        : drop_fraction_(drop_fraction), scale_kept_(scale_kept)
    {
        TORCH_CHECK(drop_fraction_ >= 0.0 && drop_fraction_ < 1.0,
                    // Cannot drop 100% this way easily without issues
                    "TargetedDropout drop_fraction must be in [0, 1).");
    }

    auto TargetedDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        vector<std::any> tensors_ = tensors;
        auto input = std::any_cast<torch::Tensor>(tensors_[0]);

        if (!this->is_training() || drop_fraction_ == 0.0)
        {
            return input;
        }

        TORCH_CHECK(input.dim() >= 1, "TargetedDropout input must be at least 1D.");

        torch::Tensor output = input.clone(); // Work on a copy

        // This dropout is typically applied per sample in a batch if input is batched.
        // We'll iterate over the batch dimension if present.
        // Or, if applied to a weight matrix, it's applied globally.
        // For simplicity, let's assume if dim > 1, dim 0 is batch.
        // And dropout is applied to features in the last dimension.

        if (input.dim() == 1)
        {
            // Single sample or global application (e.g., to flattened weights)
            apply_targeted_dropout_to_slice(output);
        }
        else
        {
            // Batched input, apply per sample along the last dimension
            for (int64_t i = 0; i < input.size(0); ++i)
            {
                // If input is (B, F), slice is (F)
                // If input is (B, S, F), slice is (S, F), need to flatten or iterate S too.
                // For simplicity, let's assume features are in the last dim, and we operate on it.
                // If input is (B, C, H, W), this gets complex.
                // This simple version will assume features are in the last dim and apply dropout there per batch item.
                // More complex targeting (e.g. per channel spatially) would need different logic.

                // This version targets units along the *last dimension* independently for each item in the first dim.
                if (input.dim() == 2)
                {
                    // (Batch, Features)
                    apply_targeted_dropout_to_slice(output[i]);
                }
                else
                {
                    // For >2D, e.g. (B, S, F), one might flatten (S,F) or apply along F for each S.
                    // Or (B, C, H, W) -> flatten (C,H,W) or operate per channel.
                    // This is a simplification: flattens all but batch dim.
                    // This means dropout is applied across all non-batch elements for each batch item.
                    int64_t batch_size = input.size(0);
                    torch::Tensor flat_input_per_batch_item = output[i].flatten();
                    apply_targeted_dropout_to_slice(flat_input_per_batch_item);
                    output[i] = flat_input_per_batch_item.view_as(output[i]); // Reshape back
                }
            }
        }
        return output;
    }

    void TargetedDropout::apply_targeted_dropout_to_slice(torch::Tensor slice)
    {
        // slice is assumed to be 1D here for simplicity of topk
        TORCH_CHECK(slice.dim() == 1, "apply_targeted_dropout_to_slice expects a 1D tensor slice.");
        if (slice.numel() == 0) return;

        int64_t num_elements_to_drop = static_cast<int64_t>(std::floor(drop_fraction_ * slice.numel()));
        if (num_elements_to_drop == 0)
        {
            return; // Nothing to drop
        }
        if (num_elements_to_drop >= slice.numel())
        {
            // Should not happen if drop_fraction < 1.0
            slice.zero_();
            return;
        }
    }
}
