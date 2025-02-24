#include <torch/torch.h>
#include <vector>
#include <memory> // For std::shared_ptr

// Function to resize a tensor
torch::Tensor resize_tensor(const torch::Tensor& input, const std::vector<int64_t>& size) {
    return torch::nn::functional::interpolate(
        input.unsqueeze(0), // Add batch dimension
        torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear)
    ).squeeze(0); // Remove batch dimension
}

int main() {
    // Define the size you want to pass to the Lambda transform
    std::vector<int64_t> size = {32, 32};

    // Create the Lambda transform with the size parameter
    auto resize_transform = torch::data::transforms::Lambda<torch::data::Example<>>(
        [size](torch::data::Example<> example) {
            example.data = resize_tensor(example.data, size);
            return example;
        }
    );

    // Example usage
    torch::Tensor sample_tensor = torch::rand({3, 64, 64}); // Example tensor (CxHxW)
    torch::data::Example<> example = {sample_tensor, 0}; // Example data and target

    // Apply the transform
    example = resize_transform(example);

    // Print the transformed tensor size
    std::cout << "Transformed Tensor Size: " << example.data.sizes() << std::endl;

    return 0;
}