#include "include/transforms/target/label_noise_injector.h"

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
    // 1. --- Setup ---
    int num_classes = 10;
    int correct_label = 7;
    std::cout << "Setup: " << num_classes << " classes, correct label is " << correct_label << "." << std::endl;

    // --- Example 1: No noise (p=0.0) ---
    std::cout << "\n--- Testing with noise_probability = 0.0 ---" << std::endl;
    xt::transforms::target::LabelNoiseInjector no_noise(num_classes, 0.0f);
    auto result1 = std::any_cast<long>(no_noise.forward({correct_label}));
    std::cout << "Output: " << result1 << " (Should always be 7)" << std::endl;

    // --- Example 2: Guaranteed noise (p=1.0) ---
    std::cout << "\n--- Testing with noise_probability = 1.0 ---" << std::endl;
    xt::transforms::target::LabelNoiseInjector all_noise(num_classes, 1.0f);
    auto result2 = std::any_cast<long>(all_noise.forward({correct_label}));
    std::cout << "Output: " << result2 << " (Should never be 7)" << std::endl;

    // --- Example 3: 50% chance of noise ---
    std::cout << "\n--- Testing with noise_probability = 0.5 (run 10 times) ---" << std::endl;
    xt::transforms::target::LabelNoiseInjector some_noise(num_classes, 0.5f);
    for (int i = 0; i < 10; ++i) {
        auto result3 = std::any_cast<long>(some_noise.forward({correct_label}));
        std::cout << "Run " << i+1 << ", Output: " << result3 << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    LabelNoiseInjector::LabelNoiseInjector(int num_classes, float noise_probability)
        : num_classes_(num_classes),
          noise_prob_(noise_probability),
          prob_dist_(0.0f, 1.0f),
          label_dist_(0, num_classes - 1) {

        if (num_classes_ <= 1) {
            throw std::invalid_argument("num_classes must be greater than 1 to inject noise.");
        }
        if (noise_prob_ < 0.0f || noise_prob_ > 1.0f) {
            throw std::invalid_argument("Noise probability must be between 0.0 and 1.0.");
        }

        // Seed the random number generator
        unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        random_engine_.seed(seed);
    }

    auto LabelNoiseInjector::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("LabelNoiseInjector::forward received an empty list.");
        }

        const std::any& input_any = any_vec[0];
        long correct_label = 0;

        if (input_any.type() == typeid(int)) {
            correct_label = std::any_cast<int>(input_any);
        } else if (input_any.type() == typeid(long)) {
            correct_label = std::any_cast<long>(input_any);
        } else {
             throw std::invalid_argument("Input to LabelNoiseInjector must be a scalar integer type.");
        }

        // 2. --- Decide whether to inject noise ---
        if (prob_dist_(random_engine_) >= noise_prob_) {
            // No noise this time, return the original label.
            return correct_label;
        }

        // 3. --- Inject Noise: Find a different label ---
        // This do-while loop is a very efficient way to guarantee we pick an
        // incorrect label. It will loop, at most, once in the vast majority of cases.
        long noisy_label;
        do {
            noisy_label = label_dist_(random_engine_);
        } while (noisy_label == correct_label);

        return noisy_label;
    }

} // namespace xt::transforms::target