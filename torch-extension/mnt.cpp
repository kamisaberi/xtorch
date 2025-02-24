#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <iostream>
#include <vector>
#include "include/datasets/mnist.h"
#include "include/models/cnn/lenet5.h"
#include <torch/data/transforms/base.h>
#include <functional>


using namespace std;
//template <typename Input, typename Output>
using Example = torch::data::Example<torch::Tensor, torch::Tensor>;

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
    std::vector<int64_t> size = {32, 32};

    auto resize_transform = torch::data::transforms::Lambda<torch::data::Example<> >(
      [size](torch::data::Example<> example) {
          example.data = resize_tensor(example.data, size);
          return example;
      }
    );

  auto normalize_transform = torch::data::transforms::Lambda<torch::data::Example<> >(
  [](torch::data::Example<> example) {
      example.data =torch::data::transforms::Normalize<>(0.5, 0.5)(example.data);
      return example;
  }
);


    std::cout.precision(10);
    torch::Device device(torch::kCPU);
    // Load the MNIST dataset


    std::vector<std::shared_ptr<torch::data::transforms::Lambda<torch::data::Example<> >>> transforms;
    transforms.push_back(std::make_shared<normalize_transform>());
//    std::vector<std::function<torch::data::Example<>>> transforms;
//    std::vector<torch::data::transforms::Transform<Input, Output>> transforms;
//    transforms.push_back(torch::data::transforms::Normalize<>(0.5,0.5));
//    transforms.push_back(resize_transform);

    auto dataset = torch::ext::data::datasets::MNIST("/home/kami/Documents/temp/",
                                                     {.mode = DataMode::TRAIN, .download = true});

    // Create a lambda function for resizing
    cout << dataset.get(0).data << endl;



    // Apply the resize transform to the dataset
    auto transformed_dataset = dataset.map(resize_transform).map(normalize_transform).map(
        torch::data::transforms::Stack<>());
    cout << transformed_dataset.get_batch(0).data << endl;

    auto transformed_dataset2 = dataset.map(resize_transform).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(
        torch::data::transforms::Stack<>());

    cout << transformed_dataset2.get_batch(0).data << endl;

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
