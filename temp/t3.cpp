#include <torch/torch.h>
#include <torch/data/transforms.h>
#include <iostream>

// Factory function that creates a scaling transform with a given factor.
torch::data::transforms::Lambda<torch::Tensor> create_scale_transform(float scale_factor) {
    return torch::data::transforms::Lambda<torch::Tensor>(
        [scale_factor](torch::Tensor input) -> torch::Tensor {
            return input * scale_factor;
        }
    );
}

int main() {
    // Create the transform by passing the scale factor as an argument.
    auto scale_transform = create_scale_transform(0.5f);

    // Create a sample tensor.
    torch::Tensor sample = torch::rand({3, 3});
    std::cout << "Original Tensor:\n" << sample << "\n\n";

    // Apply the transform.
    torch::Tensor transformed = scale_transform(sample);
    std::cout << "Transformed Tensor:\n" << transformed << "\n";

    return 0;
}
