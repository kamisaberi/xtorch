#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include "include/datasets/mnist.h"
#include "include/models/cnn/lenet5.h"

using namespace std;

void set_random() {
    torch::manual_seed(1);
    torch::cuda::manual_seed_all(1);
    srand(1);
}

// Function to resize a single tensor
torch::Tensor resize_tensor(const torch::Tensor &tensor, const std::vector<int64_t> &size) {
    return torch::nn::functional::interpolate(
        tensor.unsqueeze(0),
        torch::nn::functional::InterpolateFuncOptions().size(size).mode(torch::kBilinear).align_corners(false)
    ).squeeze(0);
}


int main() {
    std::cout.precision(10);
    torch::Device device(torch::kCPU);
    // Load the MNIST dataset
    auto dataset = torch::ext::data::datasets::MNIST("/home/kami/Documents/temp/",
                                                     {.mode = DataMode::TRAIN, .download = true});

    // Create a lambda function for resizing
    auto resize_transform = torch::data::transforms::Lambda<torch::data::Example<> >(
        [](torch::data::Example<> example) {
            example.data = resize_tensor(example.data, {32, 32});
            return example;
        }
    );

    // Apply the resize transform to the dataset
    auto transformed_dataset = dataset.map(resize_transform).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(
        torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(transformed_dataset), 64);

    torch::ext::models::LeNet5 model(10);
    model.to(device);
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));


    for (size_t epoch = 0; epoch != 10; ++epoch) {
        size_t batch_index = 0;
        auto train_loader_interator = train_loader->begin();
        auto train_loader_end = train_loader->end();
        while (train_loader_interator != train_loader_end) {
            torch::Tensor data, targets;
            auto batch = *train_loader_interator;
            data = batch.data;
            targets = batch.target;
            optimizer.zero_grad();
            torch::Tensor output;
            output = model.forward(data);
            torch::Tensor loss;
            loss = torch::nll_loss(output, targets);
            loss.backward();
            optimizer.step();
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index << " | Loss: " << loss.item<float>() <<
                        std::endl;
            }
            ++train_loader_interator;
        }
    }
    return 0;
}
