#include "include/models/computer_vision/image_classification/network_in_network.h"


using namespace std;


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // The Network-in-Network model, built from scratch for MNIST.
// struct NetworkInNetworkImpl : torch::nn::Module {
//
//     // An MLPConv block consists of a standard convolution followed by
//     // two 1x1 convolutions, which act as a small multi-layer perceptron
//     // scanning the feature maps.
//     torch::nn::Sequential mlpconv1, mlpconv2, mlpconv3;
//
//     // The final stage is a Global Average Pooling layer.
//     torch::nn::AdaptiveAvgPool2d global_avg_pool;
//
//     NetworkInNetworkImpl(int num_classes = 10)
//         // Global Average Pooling will average each feature map to a 1x1 size.
//         : global_avg_pool(torch::nn::AdaptiveAvgPool2dOptions(1))
//     {
//         // Block 1
//         mlpconv1 = torch::nn::Sequential(
//             // --- Adaptation for MNIST: Input channels is 1, not 3 ---
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 192, 5).padding(2)),
//             torch::nn::ReLU(),
//             // 1x1 Convolutions acting as the "micro-network"
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 160, 1)),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(160, 96, 1)),
//             torch::nn::ReLU(),
//             torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1)),
//             torch::nn::Dropout(0.5)
//         );
//         register_module("mlpconv1", mlpconv1);
//
//         // Block 2
//         mlpconv2 = torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 192, 5).padding(2)),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 192, 1)),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 192, 1)),
//             torch::nn::ReLU(),
//             // Using AvgPool here, as is common in later NIN-inspired architectures
//             torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(3).stride(2).padding(1)),
//             torch::nn::Dropout(0.5)
//         );
//         register_module("mlpconv2", mlpconv2);
//
//         // Block 3
//         mlpconv3 = torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 192, 3).padding(1)),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(192, 192, 1)),
//             torch::nn::ReLU(),
//             // The final 1x1 conv maps feature maps to the number of classes.
//             // This is the classification layer.
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(192, num_classes, 1)),
//             torch::nn::ReLU()
//         );
//         register_module("mlpconv3", mlpconv3);
//
//         register_module("global_avg_pool", global_avg_pool);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Pass through the MLPConv blocks
//         x = mlpconv1->forward(x);
//         x = mlpconv2->forward(x);
//         x = mlpconv3->forward(x);
//
//         // Apply Global Average Pooling.
//         // This reduces each feature map (e.g., 7x7) to a single value (1x1).
//         x = global_avg_pool->forward(x);
//
//         // Flatten the output for the loss function.
//         // The shape goes from (batch_size, num_classes, 1, 1) to (batch_size, num_classes).
//         x = x.view({x.size(0), -1});
//
//         return x;
//     }
// };
// TORCH_MODULE(NetworkInNetwork);
//
//
// // --- TRAINING AND EVALUATION LOGIC (Generic) ---
//
// template <typename DataLoader>
// void train(
//     NetworkInNetwork& model,
//     DataLoader& data_loader,
//     torch::optim::Optimizer& optimizer,
//     size_t epoch,
//     size_t dataset_size,
//     torch::Device device) {
//
//     model.train();
//     size_t batch_idx = 0;
//     for (auto& batch : data_loader) {
//         auto data = batch.data.to(device);
//         auto targets = batch.target.to(device);
//         optimizer.zero_grad();
//         auto output = model.forward(data);
//         auto loss = torch::nn::functional::cross_entropy(output, targets);
//         loss.backward();
//         optimizer.step();
//
//         if (batch_idx++ % 100 == 0) {
//             std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
//                 epoch, batch_idx * batch.data.size(0),
//                 dataset_size, loss.template item<float>());
//         }
//     }
// }
//
// template <typename DataLoader>
// void test(
//     NetworkInNetwork& model,
//     DataLoader& data_loader,
//     size_t dataset_size,
//     torch::Device device) {
//
//     torch::NoGradGuard no_grad;
//     model.eval();
//     double test_loss = 0;
//     int32_t correct = 0;
//     for (const auto& batch : data_loader) {
//         auto data = batch.data.to(device);
//         auto targets = batch.target.to(device);
//         auto output = model.forward(data);
//         test_loss += torch::nn::functional::cross_entropy(output, targets, {}, torch::Reduction::Sum).template item<double>();
//         auto pred = output.argmax(1);
//         correct += pred.eq(targets).sum().template item<int32_t>();
//     }
//
//     test_loss /= dataset_size;
//     std::printf("\nTest set: Average loss: %.4f, Accuracy: %d/%ld (%.2f%%)\n\n",
//         test_loss, correct, dataset_size,
//         100. * static_cast<double>(correct) / dataset_size);
// }
//
// // --- MAIN FUNCTION ---
//
// int main() {
//     torch::manual_seed(1);
//
//     // Hyperparameters
//     const int64_t kTrainBatchSize = 128;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 20;
//     const double kLearningRate = 0.01;
//     const double kMomentum = 0.9;
//
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training Network-in-Network on " << device << "..." << std::endl;
//
//     // Model and Optimizer
//     NetworkInNetwork model(10); // 10 classes for MNIST
//     model->to(device);
//
//     torch::optim::SGD optimizer(
//         model->parameters(),
//         torch::optim::SGDOptions(kLearningRate).momentum(kMomentum)
//     );
//
//     // Data Loaders for MNIST
//     // No need to resize images for NIN, 28x28 is fine.
//     auto train_dataset = torch::data::datasets::MNIST("./mnist_data")
//         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//         .map(torch::data::transforms::Stack<>());
//     const size_t train_dataset_size = train_dataset.size().value();
//     auto train_loader = torch::data::make_data_loader(
//         std::move(train_dataset),
//         torch::data::DataLoaderOptions().batch_size(kTrainBatchSize).workers(2)
//     );
//
//     auto test_dataset = torch::data::datasets::MNIST("./mnist_data", torch::data::datasets::MNIST::Mode::kTest)
//         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//         .map(torch::data::transforms::Stack<>());
//     const size_t test_dataset_size = test_dataset.size().value();
//     auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);
//
//     // Start Training Loop
//     for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
//         train(model, *train_loader, optimizer, epoch, train_dataset_size, device);
//         test(model, *test_loader, test_dataset_size, device);
//     }
//
//     std::cout << "Training finished." << std::endl;
//     return 0;
// }



namespace xt::models
{
    NetworkInNetwork::NetworkInNetwork(int num_classes, int in_channels)
    {
    }

    NetworkInNetwork::NetworkInNetwork(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void NetworkInNetwork::reset()
    {
    }

    auto NetworkInNetwork::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }
}
