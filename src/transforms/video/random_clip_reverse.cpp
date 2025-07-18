#include <transforms/video/random_clip_reverse.h>


#include <stdexcept>
#include <chrono>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

int main() {
    // 1. --- Create Dummy Video Clip Data ---
    // Create a simple 4-frame clip of size (4, 1, 2, 2).
    // The values will be 0, 1, 2, 3 for each frame respectively,
    // making the temporal order obvious.
    torch::Tensor clip = torch::arange(4, torch::kFloat32).view({4, 1, 1, 1}).expand({4, 1, 2, 2});

    std::cout << "--- Original Clip (sum of each frame) ---" << std::endl;
    // We print the sum of each frame to see the order clearly.
    std::cout << "[" << clip[0].sum().item<float>()
              << ", " << clip[1].sum().item<float>()
              << ", " << clip[2].sum().item<float>()
              << ", " << clip[3].sum().item<float>() << "]" << std::endl;

    // --- Example 1: Guaranteed Reversal (p=1.0) ---
    std::cout << "\n--- 1. Testing with reverse_prob = 1.0 (guaranteed reversal) ---" << std::endl;
    xt::transforms::video::RandomClipReverse reverser(1.0f);

    auto reversed_any = reverser.forward({clip});
    auto reversed_clip = std::any_cast<torch::Tensor>(reversed_any);

    std::cout << "Reversed Clip (sum of each frame):" << std::endl;
    std::cout << "[" << reversed_clip[0].sum().item<float>()
              << ", " << reversed_clip[1].sum().item<float>()
              << ", " << reversed_clip[2].sum().item<float>()
              << ", " << reversed_clip[3].sum().item<float>() << "]" << std::endl;

    // --- Example 2: Guaranteed No Reversal (p=0.0) ---
    std::cout << "\n--- 2. Testing with reverse_prob = 0.0 (never reverses) ---" << std::endl;
    xt::transforms::video::RandomClipReverse non_reverser(0.0f);

    auto non_reversed_any = non_reverser.forward({clip});
    auto non_reversed_clip = std::any_cast<torch::Tensor>(non_reversed_any);

    std::cout << "Non-Reversed Clip (sum of each frame):" << std::endl;
    std::cout << "[" << non_reversed_clip[0].sum().item<float>()
              << ", " << non_reversed_clip[1].sum().item<float>()
              << ", " << non_reversed_clip[2].sum().item<float>()
              << ", " << non_reversed_clip[3].sum().item<float>() << "]" << std::endl;

    return 0;
}
*/

namespace xt::transforms::video {

    RandomClipReverse::RandomClipReverse(float reverse_prob)
        : reverse_prob_(reverse_prob), prob_distribution_(0.0f, 1.0f) {

        if (reverse_prob_ < 0.0f || reverse_prob_ > 1.0f) {
            throw std::invalid_argument("Reverse probability must be between 0.0 and 1.0.");
        }

        // Seed the random number generator for different results on each program run.
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        random_engine_.seed(seed);
    }

    auto RandomClipReverse::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomClipReverse::forward received an empty list.");
        }

        torch::Tensor input_clip;
        try {
            input_clip = std::any_cast<torch::Tensor>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to RandomClipReverse must be a torch::Tensor.");
        }

        if (input_clip.dim() != 4) {
            throw std::invalid_argument("Input clip must be a 4D tensor (T, C, H, W).");
        }

        // 2. --- Random Check ---
        // Generate a random float and check if it's less than our probability.
        if (prob_distribution_(random_engine_) < reverse_prob_) {
            // 3. --- Apply Reversal ---
            // Use torch::flip to reverse the tensor along the first dimension (time).
            // This is highly efficient.
            return torch::flip(input_clip, {0});
        } else {
            // Otherwise, return the clip unmodified.
            return input_clip;
        }
    }

} // namespace xt::transforms::video