#include "include/dropouts/checkerboard_dropout.h"


// #include <torch/torch.h>
// #include <vector>
// #include <ostream> // For std::ostream
//
// struct CheckerboardDropoutImpl : torch::nn::Module {
//     bool drop_even_sum_indices_; // If true, drop elements where sum of relevant indices is even.
//                                  // If false, drop elements where sum of relevant indices is odd.
//
//     CheckerboardDropoutImpl(bool drop_even_sum_indices = true)
//         : drop_even_sum_indices_(drop_even_sum_indices) {}
//
//     torch::Tensor forward(const torch::Tensor& input) {
//         if (!this->is_training()) {
//             return input;
//         }
//
//         const auto input_dim = input.dim();
//         TORCH_CHECK(input_dim > 0, "CheckerboardDropout expects non-scalar input.");
//
//         torch::Tensor mask_template; // This will be 0 for dropped, 1 for kept
//
//         if (input_dim == 1) {
//             // 1D checkerboard: applies to the single dimension
//             int64_t L = input.size(0);
//             if (L == 0) return input; // Handle empty tensor
//
//             auto l_indices = torch::arange(L, input.options().dtype(torch::kLong));
//             auto characteristic_sum = l_indices % 2; // Will be 0 or 1
//
//             if (drop_even_sum_indices_) {
//                 // Drop if characteristic_sum is 0 (even), so keep if it's 1 (odd)
//                 mask_template = (characteristic_sum == 1).to(input.dtype());
//             } else {
//                 // Drop if characteristic_sum is 1 (odd), so keep if it's 0 (even)
//                 mask_template = (characteristic_sum == 0).to(input.dtype());
//             }
//         } else { // input_dim >= 2
//             // 2D+ checkerboard: applies to the last two dimensions (e.g., H, W)
//             int64_t H = input.size(-2);
//             int64_t W = input.size(-1);
//             if (H == 0 || W == 0) return input; // Handle empty spatial dimensions
//
//             auto h_indices = torch::arange(H, input.options().dtype(torch::kLong));
//             auto w_indices = torch::arange(W, input.options().dtype(torch::kLong));
//
//             // Create 2D grids for H and W indices
//             std::vector<torch::Tensor> grids = torch::meshgrid({h_indices, w_indices}, "ij"); // "ij" indexing gives HxW
//             auto h_grid = grids[0]; // Shape (H, W)
//             auto w_grid = grids[1]; // Shape (H, W)
//
//             auto characteristic_sum = (h_grid + w_grid) % 2; // Shape (H, W), values are 0 or 1
//
//             if (drop_even_sum_indices_) {
//                 // Drop if characteristic_sum is 0 (even), so keep if it's 1 (odd)
//                 mask_template = (characteristic_sum == 1).to(input.dtype());
//             } else {
//                 // Drop if characteristic_sum is 1 (odd), so keep if it's 0 (even)
//                 mask_template = (characteristic_sum == 0).to(input.dtype());
//             }
//
//             // Reshape mask_template (H,W) to be broadcastable with input (e.g., 1,1,H,W for NCHW)
//             for (int64_t i = 0; i < input_dim - 2; ++i) {
//                 mask_template = mask_template.unsqueeze(0);
//             }
//         }
//
//         // Apply the mask. Kept elements are scaled by 2.0 because 50% of elements are dropped.
//         return (input * mask_template) * 2.0;
//     }
//
//     void pretty_print(std::ostream& stream) const override {
//         stream << "CheckerboardDropout(drop_even_sum_indices="
//                << (drop_even_sum_indices_ ? "true" : "false") << ")";
//     }
// };
//
// TORCH_MODULE(CheckerboardDropout); // Creates the CheckerboardDropout module "class"
//
// /*
// // Example of how to use the CheckerboardDropout module:
// // (This is for illustration and would typically be in your main application code)
//
// #include <iostream>
//
// void run_checkerboard_dropout_example() {
//     torch::manual_seed(0); // For reproducible results
//
//     // --- Example 1: 2D-like input (e.g., a single grayscale image HxW) ---
//     CheckerboardDropout dropout_module_type0(true); // Drops elements where (h+w)%2 == 0
//     std::cout << "Checkerboard Dropout Module (drop even sum): " << dropout_module_type0 << std::endl;
//
//     torch::Tensor input_2d = torch::ones({4, 4}); // 4x4 image
//     // Expected pattern for drop_even_sum_indices = true (0 means dropped, 2 means kept and scaled):
//     // 0 2 0 2
//     // 2 0 2 0
//     // 0 2 0 2
//     // 2 0 2 0
//
//     dropout_module_type0->train(); // Set to training mode
//     torch::Tensor output_2d_train = dropout_module_type0->forward(input_2d);
//     std::cout << "Input 2D (all ones):\n" << input_2d << std::endl;
//     std::cout << "Output 2D (train, drop even sum):\n" << output_2d_train << std::endl;
//
//     dropout_module_type0->eval(); // Set to evaluation mode
//     torch::Tensor output_2d_eval = dropout_module_type0->forward(input_2d);
//     std::cout << "Output 2D (eval, drop even sum):\n" << output_2d_eval << std::endl; // Should be same as input
//
//     CheckerboardDropout dropout_module_type1(false); // Drops elements where (h+w)%2 != 0
//     std::cout << "\nCheckerboard Dropout Module (drop odd sum): " << dropout_module_type1 << std::endl;
//     // Expected pattern for drop_even_sum_indices = false (0 means dropped, 2 means kept and scaled):
//     // 2 0 2 0
//     // 0 2 0 2
//     // 2 0 2 0
//     // 0 2 0 2
//     dropout_module_type1->train();
//     output_2d_train = dropout_module_type1->forward(input_2d);
//     std::cout << "Output 2D (train, drop odd sum):\n" << output_2d_train << std::endl;
//
//
//     // --- Example 2: 4D input (e.g., NCHW) ---
//     torch::Tensor input_4d = torch::ones({1, 2, 3, 3}); // Batch=1, Channels=2, Height=3, Width=3
//     dropout_module_type0->train();
//     torch::Tensor output_4d_train = dropout_module_type0->forward(input_4d);
//     std::cout << "\nInput 4D (all ones, shape 1x2x3x3)" << std::endl;
//     std::cout << "Output 4D (train, drop even sum):\n" << output_4d_train << std::endl;
//     // Each 3x3 spatial slice within each channel should have the checkerboard pattern.
//
//
//     // --- Example 3: 1D input (e.g., a sequence) ---
//     torch::Tensor input_1d = torch::ones({5}); // Sequence of length 5
//     // Expected pattern for drop_even_sum_indices = true (0 means dropped, 2 means kept and scaled):
//     // For indices 0, 1, 2, 3, 4
//     // Sums (just i): 0, 1, 0, 1, 0
//     // Kept if sum is 1: mask [0, 1, 0, 1, 0] -> output [0, 2, 0, 2, 0]
//     dropout_module_type0->train();
//     torch::Tensor output_1d_train = dropout_module_type0->forward(input_1d);
//     std::cout << "\nInput 1D (all ones, shape 5):\n" << input_1d << std::endl;
//     std::cout << "Output 1D (train, drop even sum):\n" << output_1d_train << std::endl;
//
// }
//
// // int main() {
// //    run_checkerboard_dropout_example();
// //    return 0;
// // }
// */


namespace xt::dropouts
{
    torch::Tensor checkerboard_dropout(torch::Tensor x)
    {
        return torch::zeros(10);
    }

    auto CheckerboardDropout::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        return xt::dropouts::checkerboard_dropout(torch::zeros(10));
    }
}
