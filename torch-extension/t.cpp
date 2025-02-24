#include <torch/torch.h>
#include <vector>
#include <functional>
#include <iostream>

// Define an alias for your data sample.
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

int main() {
    // Define a vector of functions that take and return an Example.
    std::vector<std::function<Example(Example)>> transforms;

    // Create the built-in Normalize transform.
    auto normalize = torch::data::transforms::Normalize<>(0.5, 0.5);
    // Wrap Normalize in a lambda to hide its concrete type.
    transforms.push_back([normalize](Example ex) mutable -> Example {
        return normalize(std::move(ex));
    });

    // Create a built-in Lambda transform that multiplies the tensor by 2.
    auto lambda = torch::data::transforms::Lambda<Example>([](Example ex) {
        ex.data = ex.data * 2;
        return ex;
    });
    // Wrap the Lambda transform similarly.
    transforms.push_back([lambda](Example ex) mutable -> Example {
        return lambda(std::move(ex));
    });

    // Create a sample Example.
    Example sample{torch::randn({3, 224, 224}), torch::tensor(1)};

    // Apply each transform sequentially.
    for (auto &transform : transforms) {
        sample = transform(std::move(sample));
    }

    // Optionally print the resulting tensor.
    std::cout << sample.data << std::endl;

    return 0;
}
