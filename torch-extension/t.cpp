#include <torch/torch.h>
#include <vector>
#include <functional>

// Define an alias for our data sample.
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

int main() {
    // Create a vector of functions taking and returning Example.
    std::vector<std::function<Example(Example)>> transforms;

    // A normalization transform (lambda-based).
    transforms.push_back([](Example input) {
        input.data = (input.data - 0.1307) / 0.3081;
        return input;
    });

    // An example additional transform (e.g., adding random noise).
    transforms.push_back([](Example input) {
        input.data = input.data + torch::randn_like(input.data) * 0.1;
        return input;
    });

    // Create a sample example.
    Example sample{torch::randn({1, 28, 28}), torch::tensor(1)};

    // Apply each transform in sequence.
    for (auto &transform : transforms) {
        sample = transform(std::move(sample));
    }

    // Continue with further processing...
    return 0;
}
