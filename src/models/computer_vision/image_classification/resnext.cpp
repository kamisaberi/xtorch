#include "include/models/computer_vision/image_classification/resnext.h"


using namespace std;


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // --- The Core ResNeXt Block ---
//
// struct ResNeXtBlock : torch::nn::Module {
//     // A ResNeXt block splits the input into 'cardinality' groups,
//     // processes each with a small bottleneck, and then merges them.
//     // This is emented efficiently using a grouped convolution.
//     torch::nn::Conv2d conv1, conv2_grouped, conv3;
//     torch::nn::BatchNorm2d bn1, bn2, bn3;
//
//     // Shortcut connection to handle dimension changes
//     torch::nn::Sequential shortcut;
//
//     ResNeXtBlock(int in_channels, int out_channels, int stride, int cardinality, int bottleneck_width) {
//         int group_channels = cardinality * bottleneck_width;
//
//         // 1x1 convolution to enter the bottleneck
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, group_channels, 1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(group_channels));
//
//         // The core grouped convolution. This is the "split-transform" part.
//         // It has `cardinality` groups, each with `bottleneck_width` input/output channels.
//         conv2_grouped = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(group_channels, group_channels, 3)
//                 .stride(stride)
//                 .padding(1)
//                 .groups(cardinality) // The key parameter for ResNeXt
//                 .bias(false)
//         ));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(group_channels));
//
//         // 1x1 convolution to exit the bottleneck and project to the final output channels
//         conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(group_channels, out_channels, 1).bias(false)));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(out_channels));
//
//         // If dimensions change (stride > 1 or in_channels != out_channels),
//         // we need to project the shortcut connection.
//         if (stride != 1 || in_channels != out_channels) {
//             shortcut = torch::nn::Sequential(
//                 torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
//                 torch::nn::BatchNorm2d(out_channels)
//             );
//         }
//         register_module("shortcut", shortcut);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = torch::relu(bn1(conv1(x)));
//         out = torch::relu(bn2(conv2_grouped(out)));
//         out = bn3(conv3(out));
//
//         // Apply shortcut, either identity or projection
//         out += shortcut ? shortcut->forward(x) : x;
//         out = torch::relu(out);
//         return out;
//     }
// };
// TORCH_MODULE(ResNeXtBlock);
//
//
// // --- The Full ResNeXt Model ---
//
// struct ResNeXt : torch::nn::Module {
//     torch::nn::Conv2d conv1;
//     torch::nn::BatchNorm2d bn1;
//     torch::nn::Sequential layer1, layer2, layer3;
//     torch::nn::Linear linear;
//
//     ResNeXt(int num_blocks1, int num_blocks2, int num_blocks3, int cardinality, int bottleneck_width, int num_classes = 10) {
//         // --- Stem for MNIST ---
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//
//         int in_channels = 64;
//
//         // --- Stage 1 ---
//         layer1 = torch::nn::Sequential();
//         int out_channels = cardinality * bottleneck_width * 2;
//         layer1->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width));
//         in_channels = out_channels;
//         for (int i = 1; i < num_blocks1; ++i) {
//             layer1->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width));
//         }
//         register_module("layer1", layer1);
//
//         // --- Stage 2 ---
//         layer2 = torch::nn::Sequential();
//         out_channels = cardinality * bottleneck_width * 4;
//         layer2->push_back(ResNeXtBlock(in_channels, out_channels, 2, cardinality, bottleneck_width * 2)); // Downsample
//         in_channels = out_channels;
//         for (int i = 1; i < num_blocks2; ++i) {
//             layer2->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width * 2));
//         }
//         register_module("layer2", layer2);
//
//         // --- Stage 3 ---
//         layer3 = torch::nn::Sequential();
//         out_channels = cardinality * bottleneck_width * 8;
//         layer3->push_back(ResNeXtBlock(in_channels, out_channels, 2, cardinality, bottleneck_width * 4)); // Downsample
//         in_channels = out_channels;
//         for (int i = 1; i < num_blocks3; ++i) {
//             layer3->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width * 4));
//         }
//         register_module("layer3", layer3);
//
//         // --- Classifier ---
//         linear = register_module("linear", torch::nn::Linear(out_channels, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn1(conv1(x)));
//         x = layer1->forward(x);
//         x = layer2->forward(x);
//         x = layer3->forward(x);
//
//         // Global Average Pooling
//         x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));
//
//         x = x.view({x.size(0), -1});
//         x = linear->forward(x);
//         return x;
//     }
// };
// TORCH_MODULE(ResNeXt);
//
// // --- GENERIC TRAINING & TESTING FUNCTIONS ---
// template <typename DataLoader>
// void train(ResNeXt& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer,
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
// void test(ResNeXt& model, DataLoader& data_loader, size_t dataset_size, torch::Device device) {
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
//     // --- ResNeXt Hyperparameters (for a small version suitable for MNIST) ---
//     const int cardinality = 8;
//     const int bottleneck_width = 4; // Start with a small width
//     const int num_blocks1 = 2;
//     const int num_blocks2 = 2;
//     const int num_blocks3 = 2;
//
//     // --- Training Hyperparameters ---
//     const int64_t kTrainBatchSize = 128;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 20;
//     const double kLearningRate = 0.05;
//     const double kMomentum = 0.9;
//     const double kWeightDecay = 5e-4;
//
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training ResNeXt on " << device << "..." << std::endl;
//
//     // Model and Optimizer
//     ResNeXt model(num_blocks1, num_blocks2, num_blocks3, cardinality, bottleneck_width, 10);
//     model->to(device);
//
//     torch::optim::SGD optimizer(
//         model->parameters(),
//         torch::optim::SGDOptions(kLearningRate).momentum(kMomentum).weight_decay(kWeightDecay)
//     );
//
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
//         scheduler.step();
//     }
//
//     std::cout << "Training finished." << std::endl;
//     return 0;
// }


namespace xt::models
{
    // ResNeXtBlock::ResNeXtBlock(int in_channels, int out_channels, int stride, int cardinality,
    //                            int bottleneck_width)
    // {
    //     int group_channels = cardinality * bottleneck_width;
    //
    //     // 1x1 convolution to enter the bottleneck
    //     conv1 = register_module("conv1", torch::nn::Conv2d(
    //                                 torch::nn::Conv2dOptions(in_channels, group_channels, 1).bias(false)));
    //     bn1 = register_module("bn1", torch::nn::BatchNorm2d(group_channels));
    //
    //     // The core grouped convolution. This is the "split-transform" part.
    //     // It has `cardinality` groups, each with `bottleneck_width` input/output channels.
    //     conv2_grouped = register_module("conv2", torch::nn::Conv2d(
    //                                         torch::nn::Conv2dOptions(group_channels, group_channels, 3)
    //                                         .stride(stride)
    //                                         .padding(1)
    //                                         .groups(cardinality) // The key parameter for ResNeXt
    //                                         .bias(false)
    //                                     ));
    //     bn2 = register_module("bn2", torch::nn::BatchNorm2d(group_channels));
    //
    //     // 1x1 convolution to exit the bottleneck and project to the final output channels
    //     conv3 = register_module("conv3", torch::nn::Conv2d(
    //                                 torch::nn::Conv2dOptions(group_channels, out_channels, 1).bias(false)));
    //     bn3 = register_module("bn3", torch::nn::BatchNorm2d(out_channels));
    //
    //     // If dimensions change (stride > 1 or in_channels != out_channels),
    //     // we need to project the shortcut connection.
    //     if (stride != 1 || in_channels != out_channels)
    //     {
    //         shortcut = torch::nn::Sequential(
    //             torch::nn::Conv2d(
    //                 torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
    //             torch::nn::BatchNorm2d(out_channels)
    //         );
    //     }
    //     register_module("shortcut", shortcut);
    // }
    //
    // auto ResNeXtBlock::forward(std::initializer_list<std::any> tensors) -> std::any
    // {
    //     std::vector<std::any> any_vec(tensors);
    //
    //     std::vector<torch::Tensor> tensor_vec;
    //     for (const auto& item : any_vec)
    //     {
    //         tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //     }
    //
    //     torch::Tensor x = tensor_vec[0];
    //     return this->forward(x);
    // }
    //
    // torch::Tensor ResNeXtBlock::forward(torch::Tensor x)
    // {
    //     auto out = torch::relu(bn1(conv1(x)));
    //     out = torch::relu(bn2(conv2_grouped(out)));
    //     out = bn3(conv3(out));
    //
    //     // Apply shortcut, either identity or projection
    //     out += shortcut ? shortcut->forward(x) : x;
    //     out = torch::relu(out);
    //     return out;
    // }
    //
    // // --- The Full ResNeXt Model ---
    //
    // ResNeXt::ResNeXt(int num_blocks1, int num_blocks2, int num_blocks3, int cardinality, int bottleneck_width,
    //                  int num_classes)
    // {
    //     // --- Stem for MNIST ---
    //     conv1 = register_module("conv1",
    //                             torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1).bias(false)));
    //     bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
    //
    //     int in_channels = 64;
    //
    //     // --- Stage 1 ---
    //     layer1 = torch::nn::Sequential();
    //     int out_channels = cardinality * bottleneck_width * 2;
    //     layer1->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width));
    //     in_channels = out_channels;
    //     for (int i = 1; i < num_blocks1; ++i)
    //     {
    //         layer1->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width));
    //     }
    //     register_module("layer1", layer1);
    //
    //     // --- Stage 2 ---
    //     layer2 = torch::nn::Sequential();
    //     out_channels = cardinality * bottleneck_width * 4;
    //     layer2->push_back(ResNeXtBlock(in_channels, out_channels, 2, cardinality, bottleneck_width * 2)); // Downsample
    //     in_channels = out_channels;
    //     for (int i = 1; i < num_blocks2; ++i)
    //     {
    //         layer2->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width * 2));
    //     }
    //     register_module("layer2", layer2);
    //
    //     // --- Stage 3 ---
    //     layer3 = torch::nn::Sequential();
    //     out_channels = cardinality * bottleneck_width * 8;
    //     layer3->push_back(ResNeXtBlock(in_channels, out_channels, 2, cardinality, bottleneck_width * 4)); // Downsample
    //     in_channels = out_channels;
    //     for (int i = 1; i < num_blocks3; ++i)
    //     {
    //         layer3->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width * 4));
    //     }
    //     register_module("layer3", layer3);
    //
    //     // --- Classifier ---
    //     linear = register_module("linear", torch::nn::Linear(out_channels, num_classes));
    // }
    //
    // auto ResNeXt::forward(std::initializer_list<std::any> tensors) -> std::any
    // {
    //     std::vector<std::any> any_vec(tensors);
    //
    //     std::vector<torch::Tensor> tensor_vec;
    //     for (const auto& item : any_vec)
    //     {
    //         tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //     }
    //
    //     torch::Tensor x = tensor_vec[0];
    //     return this->forward(x);
    // }
    //
    // torch::Tensor ResNeXt::forward(torch::Tensor x)
    // {
    //     x = torch::relu(bn1(conv1(x)));
    //     x = layer1->forward(x);
    //     x = layer2->forward(x);
    //     x = layer3->forward(x);
    //
    //     // Global Average Pooling
    //     x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));
    //
    //     x = x.view({x.size(0), -1});
    //     x = linear->forward(x);
    //     return x;
    // }

    //    ResNeXt::ResNeXt(int num_classes, int in_channels)
    //    {
    //    }
    //
    //    ResNeXt::ResNeXt(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    //    {
    //    }
    //
    //    void ResNeXt::reset()
    //    {
    //    }
    //
    //    auto ResNeXt::forward(std::initializer_list<std::any> tensors) -> std::any
    //    {
    //        std::vector<std::any> any_vec(tensors);
    //
    //        std::vector<torch::Tensor> tensor_vec;
    //        for (const auto& item : any_vec)
    //        {
    //            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
    //        }
    //
    //        torch::Tensor x = tensor_vec[0];
    //
    //        return x;
    //    }
}
