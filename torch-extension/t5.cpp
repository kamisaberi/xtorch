#include <torch/torch.h>
#include <torch/data/datasets.h>
#include <torch/data/transforms.h>
#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include "include/datasets/mnist.h"
#include "include/models/cnn/lenet5.h"
#include <torch/data/transforms/base.h>
#include <functional>
#include "include/definitions/transforms.h"



int main() {
    // Load the MNIST dataset
    auto dataset = torch::data::datasets::MNIST("data/MNIST/raw");

    // Define a vector of transformations
    std::vector<torch::data::transforms::Lambda<torch::data::Example<>>> transforms = {
        torch::data::transforms::Lambda([](torch::data::Example<> example) {
            // Resize transformation
            auto resize = torch::ext::data::transforms::resize({32, 32});
            example.data = resize(example.data);
            return example;
        }),
        torch::data::transforms::Lambda([](torch::data::Example<> example) {
            // Normalize transformation
            auto normalize = torch::ext::data::transforms::normalize(0.5, 0.5);
            example.data = normalize(example.data);
            return example;
        })
    };

    // Apply transformations sequentially using map()
    for (const auto& transform : transforms) {
        dataset = dataset.map(transform);
    }

    // Create a data loader for the transformed dataset
    auto data_loader = torch::data::make_data_loader(
        std::move(dataset),
        torch::data::DataLoaderOptions().batch_size(64).workers(2)
    );

    // Iterate over the data loader and print the shapes of the batches
    for (auto& batch : *data_loader) {
        std::cout << "Transformed data shape: " << batch.data()->data << std::endl;
        std::cout << "Transformed target shape: " << batch.data()->target << std::endl;
        break; // Print only the first batch for demonstration
    }

    return 0;
}