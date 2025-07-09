#include "include/models/computer_vision/image_classification/se_net.h"


using namespace std;


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // --- The Core Squeeze-and-Excitation Module ---
//
// struct SEModuleImpl : torch::nn::Module {
//     torch::nn::AdaptiveAvgPool2d squeeze;
//     torch::nn::Sequential excitation;
//
//     SEModuleImpl(int in_channels, int reduction_ratio = 16)
//         : squeeze(torch::nn::AdaptiveAvgPool2dOptions(1))
//     {
//         int reduced_channels = in_channels / reduction_ratio;
//         // Use 1x1 Convs to act as fully connected layers
//         excitation = torch::nn::Sequential(
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, reduced_channels, 1)),
//             torch::nn::ReLU(),
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(reduced_channels, in_channels, 1)),
//             torch::nn::Sigmoid()
//         );
//         register_module("squeeze", squeeze);
//         register_module("excitation", excitation);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // Squeeze: Global information embedding
//         auto squeezed = squeeze(x);
//         // Excitation: Learn channel-wise weights
//         auto weights = excitation->forward(squeezed);
//         // Rescale: Apply weights to original feature maps
//         return x * weights;
//     }
// };
// TORCH_MODULE(SEModule);
//
//
// // --- A Basic ResNet Block with an integrated SE Module ---
//
// struct SEBasicBlockImpl : torch::nn::Module {
//     torch::nn::Conv2d conv1, conv2;
//     torch::nn::BatchNorm2d bn1, bn2;
//     SEModule se_module;
//
//     // Shortcut for residual connection
//     torch::nn::Sequential shortcut;
//
//     SEBasicBlockImpl(int in_planes, int planes, int stride = 1, int reduction_ratio = 16)
//         : conv1(torch::nn::Conv2dOptions(in_planes, planes, 3).stride(stride).padding(1).bias(false)),
//           bn1(planes),
//           conv2(torch::nn::Conv2dOptions(planes, planes, 3).stride(1).padding(1).bias(false)),
//           bn2(planes),
//           se_module(planes, reduction_ratio)
//     {
//         register_module("conv1", conv1);
//         register_module("bn1", bn1);
//         register_module("conv2", conv2);
//         register_module("bn2", bn2);
//         register_module("se_module", se_module);
//
//         // If dimensions change, project the shortcut
//         if (stride != 1 || in_planes != planes) {
//             shortcut = torch::nn::Sequential(
//                 torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1).stride(stride).bias(false)),
//                 torch::nn::BatchNorm2d(planes)
//             );
//             register_module("shortcut", shortcut);
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = torch::relu(bn1(conv1(x)));
//         out = bn2(conv2(out));
//
//         // Apply Squeeze-and-Excitation here!
//         out = se_module->forward(out);
//
//         // Add shortcut
//         auto short_x = shortcut ? shortcut->forward(x) : x;
//         out += short_x;
//
//         out = torch::relu(out);
//         return out;
//     }
// };
// TORCH_MODULE(SEBasicBlock);
//
//
// // --- The Full SENet Model (using the SE-ResNet backbone) ---
//
// struct SENetImpl : torch::nn::Module {
//     torch::nn::Conv2d conv1;
//     torch::nn::BatchNorm2d bn1;
//     torch::nn::Sequential layer1, layer2, layer3;
//     torch::nn::Linear linear;
//
//     SENetImpl(const std::vector<int>& num_blocks, int num_classes = 10, int reduction_ratio = 16) {
//         // Stem for MNIST (1 input channel)
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//
//         int in_planes = 64;
//         layer1 = _make_layer(in_planes, 64, num_blocks[0], 1, reduction_ratio);
//         layer2 = _make_layer(64, 128, num_blocks[1], 2, reduction_ratio); // Downsample
//         layer3 = _make_layer(128, 256, num_blocks[2], 2, reduction_ratio); // Downsample
//
//         register_module("layer1", layer1);
//         register_module("layer2", layer2);
//         register_module("layer3", layer3);
//
//         linear = register_module("linear", torch::nn::Linear(256, num_classes));
//     }
//
//     torch::nn::Sequential _make_layer(int& in_planes, int planes, int num_blocks, int stride, int reduction) {
//         torch::nn::Sequential layers;
//         layers->push_back(SEBasicBlock(in_planes, planes, stride, reduction));
//         in_planes = planes; // Update in_planes for the next block
//         for(int i = 1; i < num_blocks; ++i) {
//             layers->push_back(SEBasicBlock(in_planes, planes, 1, reduction));
//         }
//         return layers;
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
// TORCH_MODULE(SENet);
//
// // --- GENERIC TRAINING & TESTING FUNCTIONS ---
// template <typename DataLoader>
// void train(SENet& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer,
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
// void test(SENet& model, DataLoader& data_loader, size_t dataset_size, torch::Device device) {
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
//     // --- Training Hyperparameters ---
//     const int64_t kTrainBatchSize = 128;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 15;
//     const double kLearningRate = 0.1;
//     const double kMomentum = 0.9;
//     const double kWeightDecay = 5e-4;
//
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training SENet on " << device << "..." << std::endl;
//
//     // --- Model Configuration ---
//     // A small SE-ResNet for MNIST: 3 stages with 2 blocks each
//     std::vector<int> num_blocks = {2, 2, 2};
//     SENet model(num_blocks);
//     model->to(device);
//
//     torch::optim::SGD optimizer(
//         model->parameters(),
//         torch::optim::SGDOptions(kLearningRate).momentum(kMomentum).weight_decay(kWeightDecay)
//     );
//
//     auto scheduler = torch::optim::StepLR(optimizer, /*step_size=*/5, /*gamma=*/0.1);
//
//     // --- Data Loading ---
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
//     // --- Training Loop ---
//     for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
//         train(model, *train_loader, optimizer, epoch, train_dataset_size, device);
//         test(model, *test_loader, test_dataset_size, device);
//         scheduler.step();
//     }
//
//     std::cout << "Training finished." << std::endl;
//     return 0;
// }


namespace xt::models {


    SEModuleImpl::SEModuleImpl(int in_channels, int reduction_ratio = 16)
            : squeeze(torch::nn::AdaptiveAvgPool2dOptions(1)) {
        int reduced_channels = in_channels / reduction_ratio;
        // Use 1x1 Convs to act as fully connected layers
        excitation = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, reduced_channels, 1)),
                torch::nn::ReLU(),
                torch::nn::Conv2d(torch::nn::Conv2dOptions(reduced_channels, in_channels, 1)),
                torch::nn::Sigmoid()
        );
        register_module("squeeze", squeeze);
        register_module("excitation", excitation);
    }

    torch::Tensor SEModuleImpl::forward(torch::Tensor x) {
        // Squeeze: Global information embedding
        auto squeezed = squeeze(x);
        // Excitation: Learn channel-wise weights
        auto weights = excitation->forward(squeezed);
        // Rescale: Apply weights to original feature maps
        return x * weights;
    }


    // --- A Basic ResNet Block with an integrated SE Module ---

        SEBasicBlockImpl::SEBasicBlockImpl(int in_planes, int planes, int stride = 1, int reduction_ratio = 16)
                : conv1(torch::nn::Conv2dOptions(in_planes, planes, 3).stride(stride).padding(1).bias(false)),
                  bn1(planes),
                  conv2(torch::nn::Conv2dOptions(planes, planes, 3).stride(1).padding(1).bias(false)),
                  bn2(planes),
                  se_module(planes, reduction_ratio) {
            register_module("conv1", conv1);
            register_module("bn1", bn1);
            register_module("conv2", conv2);
            register_module("bn2", bn2);
            register_module("se_module", se_module);

            // If dimensions change, project the shortcut
            if (stride != 1 || in_planes != planes) {
                shortcut = torch::nn::Sequential(
                        torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1).stride(stride).bias(false)),
                        torch::nn::BatchNorm2d(planes)
                );
                register_module("shortcut", shortcut);
            }
        }

        torch::Tensor SEBasicBlockImpl::forward(torch::Tensor x) {
            auto out = torch::relu(bn1(conv1(x)));
            out = bn2(conv2(out));

            // Apply Squeeze-and-Excitation here!
            out = se_module->forward(out);

            // Add shortcut
            auto short_x = shortcut ? shortcut->forward(x) : x;
            out += short_x;

            out = torch::relu(out);
            return out;
        }

    // --- The Full SENet Model (using the SE-ResNet backbone) ---

        SENetImpl::SENetImpl(const std::vector<int> &num_blocks, int num_classes = 10, int reduction_ratio = 16) {
            // Stem for MNIST (1 input channel)
            conv1 = register_module("conv1", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1).bias(false)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

            int in_planes = 64;
            layer1 = _make_layer(in_planes, 64, num_blocks[0], 1, reduction_ratio);
            layer2 = _make_layer(64, 128, num_blocks[1], 2, reduction_ratio); // Downsample
            layer3 = _make_layer(128, 256, num_blocks[2], 2, reduction_ratio); // Downsample

            register_module("layer1", layer1);
            register_module("layer2", layer2);
            register_module("layer3", layer3);

            linear = register_module("linear", torch::nn::Linear(256, num_classes));
        }

        torch::nn::Sequential SENetImpl::_make_layer(int &in_planes, int planes, int num_blocks, int stride, int reduction) {
            torch::nn::Sequential layers;
            layers->push_back(SEBasicBlock(in_planes, planes, stride, reduction));
            in_planes = planes; // Update in_planes for the next block
            for (int i = 1; i < num_blocks; ++i) {
                layers->push_back(SEBasicBlock(in_planes, planes, 1, reduction));
            }
            return layers;
        }

        torch::Tensor SENetImpl::forward(torch::Tensor x) {
            x = torch::relu(bn1(conv1(x)));
            x = layer1->forward(x);
            x = layer2->forward(x);
            x = layer3->forward(x);

            // Global Average Pooling
            x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));

            x = x.view({x.size(0), -1});
            x = linear->forward(x);
            return x;
        }

//    SENet::SENet(int num_classes, int in_channels)
//    {
//    }
//
//    SENet::SENet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
//    {
//    }
//
//    void SENet::reset()
//    {
//    }
//
//    auto SENet::forward(std::initializer_list<std::any> tensors) -> std::any
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
