#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <torch/data/transforms.h>
#include <iostream>

int main() {
    // Load the MNIST dataset and apply a sequence of lambda transforms.
    // Each map() call applies a lambda that takes a data sample and returns
    // a modified data sample. The lambda functions are not part of the final output.
    auto dataset = torch::data::datasets::MNIST("./data")
        // First transform: normalize the image data to [0, 1].
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(
            [](torch::data::Example<> example) -> torch::data::Example<> {
                example.data = example.data.to(torch::kFloat32).div(255.0);
                return example;
            }
        ))
        // Second transform: add a constant bias (e.g., add 1 to each pixel).
        .map(torch::data::transforms::Lambda<torch::data::Example<>>(
            [](torch::data::Example<> example) -> torch::data::Example<> {
                example.data = example.data + 1;
                return example;
            }
        ));

    // When you access a sample from the dataset, only the transformed data and target are returned.
    auto transformed_example = dataset.get(0);
    std::cout << "Transformed Data:\n" << transformed_example.data << "\n";
    std::cout << "Target:\n" << transformed_example.target << "\n";

    return 0;
}
