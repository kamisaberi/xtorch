#include "include/transforms/weather/accumulated_snow.h"


// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>

int main() {
    // 1. Create a dummy tensor representing daily snowfall in cm for a week.
    torch::Tensor daily_snowfall = torch::tensor({2.5, 10.0, 0.0, 5.5, 0.0, 1.2, 8.0});
    std::cout << "Daily Snowfall (cm):\n" << daily_snowfall << std::endl;

    // 2. Create and apply the transform.
    // We want to accumulate along the single dimension (dim=0).
    xt::transforms::weather::AccumulatedSnow accumulator(0);
    torch::Tensor total_snow = std::any_cast<torch::Tensor>(accumulator.forward({daily_snowfall}));

    // 3. Print the result.
    std::cout << "\nAccumulated Snow (cm) over the week:\n" << total_snow << std::endl;

    // --- Expected Output ---
    // Daily Snowfall (cm):
    //  2.5000
    // 10.0000
    //  0.0000
    //  5.5000
    //  0.0000
    //  1.2000
    //  8.0000
    // [ CPUFloatType{7} ]
    //
    // Accumulated Snow (cm) over the week:
    //  2.5000
    // 12.5000
    // 12.5000
    // 18.0000
    // 18.0000
    // 19.2000
    // 27.2000
    // [ CPUFloatType{7} ]

    return 0;
}
*/

namespace xt::transforms::weather {

    AccumulatedSnow::AccumulatedSnow() : dim_(0) {}

    AccumulatedSnow::AccumulatedSnow(int dim) : dim_(dim) {}

    auto AccumulatedSnow::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("AccumulatedSnow::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to AccumulatedSnow is not defined.");
        }

        // Check if the dimension is valid for the given tensor.
        if (dim_ >= input_tensor.dim() || dim_ < -input_tensor.dim()) {
            throw std::out_of_range("Accumulation dimension is out of range for the input tensor.");
        }

        // 2. --- Apply Accumulation ---
        // Use torch::cumsum to perform the accumulation efficiently.
        // This function is highly optimized for this exact operation.
        torch::Tensor accumulated_tensor = torch::cumsum(input_tensor, dim_);

        // 3. --- Return Result ---
        return accumulated_tensor;
    }

} // namespace xt::transforms::weather