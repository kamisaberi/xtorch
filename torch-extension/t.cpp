#include <torch/torch.h>
#include <vector>
#include <memory> // For std::shared_ptr

int main() {
    // Define a vector to hold shared pointers to Normalize objects
    std::vector<std::shared_ptr<torch::data::transforms::Normalize<>>> transforms;

    // Add a Normalize transform to the vector
    transforms.push_back(
        std::make_shared<torch::data::transforms::Normalize<>>(0.5, 0.5)
    );

    // Apply the transform to a sample tensor
    at::Tensor sample = torch::ones({3, 3}); // Example tensor
    for (const auto& transform : transforms) {
        sample = transform->apply(sample);
    }

    // Print the transformed tensor
    std::cout << "Transformed Tensor:\n" << sample << std::endl;

    return 0;
}