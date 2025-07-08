#include "include/models/computer_vision/image_classification/pyramidal_net.h"


using namespace std;

//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <cmath>
//
// // --- The Core Pyramidal Residual Block ---
//
// struct PyramidalBasicBlockImpl : torch::nn::Module {
//     torch::nn::Conv2d conv1{nullptr}, conv2_nobn{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
//
//     int stride = 1;
//
//     PyramidalBasicBlockImpl(int in_planes, int out_planes, int stride = 1)
//         : stride(stride)
//     {
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, 3).stride(stride).padding(1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_planes));
//
//         // The second conv maps back to the same number of output planes
//         conv2_nobn = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(out_planes, out_planes, 3).stride(1).padding(1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_planes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = torch::relu(bn1(conv1(x)));
//         out = bn2(conv2_nobn(out));
//
//         // --- The Pyramidal Shortcut Connection ---
//         // This is the key part of the architecture.
//         // If dimensions change, we use a zero-padding shortcut.
//         torch::Tensor shortcut = x;
//         int in_channels = x.size(1);
//         int out_channels = out.size(1);
//
//         if (stride != 1 || in_channels != out_channels) {
//             // Downsample spatially if stride is not 1
//             if (stride != 1) {
//                 shortcut = torch::nn::functional::avg_pool2d(shortcut,
//                     torch::nn::functional::AvgPool2dFuncOptions(2));
//             }
//
//             // Pad with zero-channels to match output dimensions
//             int64_t pad_channels = out_channels - in_channels;
//             if (pad_channels > 0) {
//                 auto padding = torch::zeros({shortcut.size(0), pad_channels, shortcut.size(2), shortcut.size(3)}, shortcut.options());
//                 shortcut = torch::cat({shortcut, padding}, 1);
//             }
//         }
//
//         out += shortcut;
//         out = torch::relu(out);
//         return out;
//     }
// };
// TORCH_MODULE(PyramidalBasicBlock);
//
//
// // --- The Full PyramidalNet Model ---
//
// struct PyramidalNetImpl : torch::nn::Module {
//     torch::nn::Conv2d conv1;
//     torch::nn::BatchNorm2d bn1;
//     torch::nn::Sequential layer1, layer2, layer3;
//     torch::nn::Linear linear;
//
//     PyramidalNetImpl(int N, int alpha, int num_classes = 10)
//     {
//         // For MNIST, we use a simple stem
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(16));
//
//         // Calculate how many channels to add at each block
//         const double add_per_block = static_cast<double>(alpha) / N;
//         double current_planes = 16.0;
//
//         // Stage 1
//         layer1 = torch::nn::Sequential();
//         for (int i = 0; i < N; ++i) {
//             int in_planes = static_cast<int>(round(current_planes));
//             current_planes += add_per_block;
//             int out_planes = static_cast<int>(round(current_planes));
//             layer1->push_back(PyramidalBasicBlock(in_planes, out_planes, 1));
//         }
//         register_module("layer1", layer1);
//
//         // Stage 2
//         layer2 = torch::nn::Sequential();
//         for (int i = 0; i < N; ++i) {
//             int stride = (i == 0) ? 2 : 1; // Downsample at the start of the stage
//             int in_planes = static_cast<int>(round(current_planes));
//             current_planes += add_per_block;
//             int out_planes = static_cast<int>(round(current_planes));
//             layer2->push_back(PyramidalBasicBlock(in_planes, out_planes, stride));
//         }
//         register_module("layer2", layer2);
//
//         // Stage 3
//         layer3 = torch::nn::Sequential();
//         for (int i = 0; i < N; ++i) {
//             int stride = (i == 0) ? 2 : 1;
//             int in_planes = static_cast<int>(round(current_planes));
//             current_planes += add_per_block;
//             int out_planes = static_cast<int>(round(current_planes));
//             layer3->push_back(PyramidalBasicBlock(in_planes, out_planes, stride));
//         }
//         register_module("layer3", layer3);
//
//         // Final classifier
//         int final_planes = static_cast<int>(round(current_planes));
//         linear = register_module("linear", torch::nn::Linear(final_planes, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn1(conv1(x)));
//         x = layer1->forward(x);
//         x = layer2->forward(x);
//         x = layer3->forward(x);
//
//         // Global Average Pooling
//         x = torch::nn::functional::adaptive_avg_pool2d(x,
//             torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));
//
//         x = x.view({x.size(0), -1});
//         x = linear->forward(x);
//         return x;
//     }
// };
// TORCH_MODULE(PyramidalNet);
//
// // --- GENERIC TRAINING & TESTING FUNCTIONS ---
// template <typename DataLoader>
// void train(PyramidalNet& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer,
//            size_t epoch, size_t dataset_size, torch::Device device) {
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
//         if (batch_idx++ % 100 == 0) {
//             std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
//                 epoch, batch_idx * batch.data.size(0), dataset_size, loss.template item<float>());
//         }
//     }
// }
//
// template <typename DataLoader>
// void test(PyramidalNet& model, DataLoader& data_loader, size_t dataset_size, torch::Device device) {
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
//     test_loss /= dataset_size;
//     std::printf("\nTest set: Average loss: %.4f, Accuracy: %d/%ld (%.2f%%)\n\n",
//         test_loss, correct, dataset_size, 100. * static_cast<double>(correct) / dataset_size);
// }
//
// // --- MAIN FUNCTION ---
// int main() {
//     torch::manual_seed(1);
//
//     // --- PyramidalNet Hyperparameters ---
//     // N = number of blocks per stage. Total depth is roughly 1 (stem) + 3*N*2 (blocks) + 1 (fc).
//     const int N = 6;
//     // alpha = total number of channels to add over the whole network.
//     const int alpha = 48;
//
//     // --- Training Hyperparameters ---
//     const int64_t kTrainBatchSize = 128;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 20;
//     const double kLearningRate = 0.1;
//     const double kMomentum = 0.9;
//     const double kWeightDecay = 1e-4;
//
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training PyramidalNet on " << device << "..." << std::endl;
//     std::cout << "Model depth: ~" << (1 + 3 * N * 2 + 1) << " layers, alpha: " << alpha << std::endl;
//
//     // Model and Optimizer
//     PyramidalNet model(N, alpha, 10);
//     model->to(device);
//
//     torch::optim::SGD optimizer(
//         model->parameters(),
//         torch::optim::SGDOptions(kLearningRate).momentum(kMomentum).weight_decay(kWeightDecay)
//     );
//
//     // Learning Rate Scheduler
//     auto scheduler = torch::optim::StepLR(optimizer, /*step_size=*/8, /*gamma=*/0.1);
//
//     // Data Loaders for MNIST
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
//         scheduler.step(); // Update learning rate
//     }
//
//     std::cout << "Training finished." << std::endl;
//     return 0;
// }



namespace xt::models
{
    PyramidalNet::PyramidalNet(int num_classes, int in_channels)
    {
    }

    PyramidalNet::PyramidalNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PyramidalNet::reset()
    {
    }

    auto PyramidalNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
