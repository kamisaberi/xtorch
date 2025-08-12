#include <torch/torch.h>
#include <torch/data/transforms.h>
#include <vector>
#include <iostream>

int main() {
    // Define a lambda transform to normalize a tensor (e.g., scale pixel values to [0, 1])
    auto normalize = torch::data::transforms::Lambda<torch::Tensor>(
        [](torch::Tensor input) -> torch::Tensor {
            return input.to(torch::kFloat32).div(255);
        }
    );

    // Define another lambda transform that adds a constant bias (e.g., adds 1)
    auto add_bias = torch::data::transforms::Lambda<torch::Tensor>(
        [](torch::Tensor input) -> torch::Tensor {
            return input + 1;
        }
    );

    // Create a vector to hold the lambda transforms.
    // Since both transforms operate on torch::Tensor, they have the same type.
    std::vector<torch::data::transforms::Lambda<torch::Tensor>> transforms;
    transforms.push_back(normalize);
    transforms.push_back(add_bias);

    // Create a sample tensor (for example, a random image tensor with values in [0, 255])
    torch::Tensor sample = torch::randint(0, 256, {1, 3, 28, 28});

    // Apply each transform in sequence from the vector.
    for (auto& transform : transforms) {
        sample = transform(sample);
    }

    // Output the resulting tensor
    std::cout << sample << std::endl;

    return 0;
}
