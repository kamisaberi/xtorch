#include <torch/torch.h>
#include <iostream>

int main() {
    // Create a CrossEntropyLoss criterion
    torch::nn::CrossEntropyLoss criterion;

    // Example input (logits) and target (labels)
    // logits should have shape [batch_size, num_classes]
    torch::Tensor logits = torch::rand({3, 5}, torch::kFloat); // Example logits for 3 samples and 5 classes
    std::cout << logits.sizes() << std::endl;
    torch::Tensor target = torch::tensor({1, 0, 4}, torch::kLong); // Example targets (class indices)

    std::cout << target.sizes() << std::endl;
    // Compute the loss
    torch::Tensor loss = criterion(logits, target);

    // Print the loss
    std::cout << "Loss: " << loss.item<float>() << std::endl;

    return 0;
}
