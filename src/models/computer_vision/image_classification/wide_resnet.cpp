#include "include/models/computer_vision/image_classification/wide_resnet.h"


using namespace std;

// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // --- The Core Wide-ResNet Block ---
// // This block is different from a standard ResNet block. It uses a (BN-ReLU-Conv) pre-activation
// // sequence and includes a Dropout layer.
//
// struct WideBasicBlockImpl : torch::nn::Module {
//     torch::nn::BatchNorm2d bn1, bn2;
//     torch::nn::Conv2d conv1, conv2;
//     torch::nn::Dropout dropout;
//
//     // Shortcut connection for residual path
//     torch::nn::Sequential shortcut;
//
//     WideBasicBlockImpl(int in_planes, int planes, double dropout_rate, int stride = 1)
//         : bn1(in_planes),
//           conv1(torch::nn::Conv2dOptions(in_planes, planes, 3).stride(stride).padding(1).bias(false)),
//           bn2(planes),
//           conv2(torch::nn::Conv2dOptions(planes, planes, 3).stride(1).padding(1).bias(false)),
//           dropout(dropout_rate)
//     {
//         register_module("bn1", bn1);
//         register_module("conv1", conv1);
//         register_module("bn2", bn2);
//         register_module("conv2", conv2);
//         register_module("dropout", dropout);
//
//         // If dimensions change, we need to project the shortcut with a 1x1 conv
//         if (stride != 1 || in_planes != planes) {
//             shortcut = torch::nn::Sequential(
//                 torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1).stride(stride).bias(false))
//             );
//             register_module("shortcut", shortcut);
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = torch::relu(bn1(x));
//         out = conv1(out);
//         out = dropout(out); // Dropout is applied here in the WideResNet block
//         out = torch::relu(bn2(out));
//         out = conv2(out);
//
//         auto short_x = shortcut ? shortcut->forward(x) : x;
//         out += short_x;
//
//         return out;
//     }
// };
// TORCH_MODULE(WideBasicBlock);
//
//
// // --- The Full WideResNet Model ---
//
// struct WideResNetImpl : torch::nn::Module {
//     torch::nn::Conv2d conv1;
//     torch::nn::Sequential layer1, layer2, layer3;
//     torch::nn::BatchNorm2d bn_final;
//     torch::nn::Linear linear;
//
//     WideResNetImpl(int depth, int widen_factor, double dropout_rate, int num_classes = 10) {
//         // Depth must be of the form 6*N + 4
//         assert((depth - 4) % 6 == 0 && "WideResNet depth should be 6n+4");
//         int N = (depth - 4) / 6;
//
//         // --- Stem (for MNIST, 1 input channel) ---
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1).bias(false)));
//
//         int in_planes = 16;
//
//         // --- Stage 1 ---
//         int out_planes1 = 16 * widen_factor;
//         layer1 = _make_layer(in_planes, out_planes1, N, 1, dropout_rate);
//         in_planes = out_planes1;
//
//         // --- Stage 2 ---
//         int out_planes2 = 32 * widen_factor;
//         layer2 = _make_layer(in_planes, out_planes2, N, 2, dropout_rate); // Downsample
//         in_planes = out_planes2;
//
//         // --- Stage 3 ---
//         int out_planes3 = 64 * widen_factor;
//         layer3 = _make_layer(in_planes, out_planes3, N, 2, dropout_rate); // Downsample
//         in_planes = out_planes3;
//
//         register_module("layer1", layer1);
//         register_module("layer2", layer2);
//         register_module("layer3", layer3);
//
//         // --- Classifier ---
//         bn_final = register_module("bn_final", torch::nn::BatchNorm2d(in_planes));
//         linear = register_module("linear", torch::nn::Linear(in_planes, num_classes));
//     }
//
//     torch::nn::Sequential _make_layer(int in_planes, int planes, int num_blocks, int stride, double dropout_rate) {
//         torch::nn::Sequential layers;
//         layers->push_back(WideBasicBlock(in_planes, planes, dropout_rate, stride));
//         for (int i = 1; i < num_blocks; ++i) {
//             layers->push_back(WideBasicBlock(planes, planes, dropout_rate, 1));
//         }
//         return layers;
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = conv1(x);
//         x = layer1->forward(x);
//         x = layer2->forward(x);
//         x = layer3->forward(x);
//
//         x = torch::relu(bn_final(x));
//
//         // Global Average Pooling
//         x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));
//
//         x = x.view({x.size(0), -1});
//         x = linear->forward(x);
//         return x;
//     }
// };
// TORCH_MODULE(WideResNet);
//
// // --- GENERIC TRAINING & TESTING FUNCTIONS ---
// template <typename DataLoader>
// void train(WideResNet& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer,
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
// void test(WideResNet& model, DataLoader& data_loader, size_t dataset_size, torch::Device device) {
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
//     // --- WideResNet Hyperparameters ---
//     // A small WRN-16-4 for MNIST. 16 layers deep, widen_factor=4
//     const int depth = 16;
//     const int widen_factor = 4;
//     const double dropout_rate = 0.3;
//
//     // --- Training Hyperparameters ---
//     const int64_t kTrainBatchSize = 128;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 20;
//     const double kLearningRate = 0.1;
//     const double kMomentum = 0.9;
//     const double kWeightDecay = 5e-4;
//
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training WideResNet-" << depth << "-" << widen_factor << " on " << device << "..." << std::endl;
//
//     // Model and Optimizer
//     WideResNet model(depth, widen_factor, dropout_rate, 10);
//     model->to(device);
//
//     torch::optim::SGD optimizer(
//         model->parameters(),
//         torch::optim::SGDOptions(kLearningRate).momentum(kMomentum).weight_decay(kWeightDecay)
//     );
//
//     // Learning rate scheduler that decays LR at epochs 10 and 15
//     auto scheduler = torch::optim::MultiStepLR(optimizer, /*milestones=*/{10, 15}, /*gamma=*/0.1);
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
//     // Training Loop
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
    WideResNet::WideResNet(int num_classes, int in_channels)
    {
    }

    WideResNet::WideResNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void WideResNet::reset()
    {
    }

    auto WideResNet::forward(std::initializer_list<std::any> tensors) -> std::any
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


    // namespace {
    //     ResidualBlock::ResidualBlock(int in_channels, int out_channels, int stride, torch::nn::Sequential downsample) {
    //         conv1 = torch::nn::Sequential(
    //             torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)),
    //             torch::nn::BatchNorm2d(out_channels),
    //             torch::nn::ReLU()
    //         );
    //
    //         conv2 = torch::nn::Sequential(
    //             torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1)),
    //             torch::nn::BatchNorm2d(out_channels)
    //         );
    //
    //         register_module("conv1", conv1);
    //         register_module("conv2", conv2);
    //
    //
    //         this->downsample = downsample;
    //         this->relu = torch::nn::ReLU();
    //         this->out_channels = out_channels;
    //     }
    //
    //     torch::Tensor ResidualBlock::forward(torch::Tensor x) {
    //         residual = x;
    //         torch::Tensor out = conv1->forward(x);
    //         out = conv2->forward(out);
    //         if (downsample) {
    //             residual = downsample->forward(x);
    //         } else {
    //         }
    //         out += residual;
    //         out = relu(out);
    //         return out;
    //     }
    // }
    //
    //
    // WideResNet::WideResNet(vector<int> layers, int num_classes, int in_channels) : BaseModel() {
    //     inplanes = 64;
    //
    //
    //     conv1 = torch::nn::Sequential(
    //         torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
    //         torch::nn::BatchNorm2d(64),
    //         torch::nn::ReLU()
    //     );
    //
    //     register_module("conv1", conv1);
    //
    //     maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
    //     // maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
    //
    //     layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
    //     layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
    //     layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
    //     layer3 = makeLayerFromResidualBlock(512, layers[3], 2);
    //
    //     register_module("layer0", layer0);
    //     register_module("layer1", layer1);
    //     register_module("layer2", layer2);
    //     register_module("layer3", layer3);
    //
    //     avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
    //     fc = torch::nn::Linear(512, num_classes);
    // }
    //
    //
    // WideResNet::WideResNet(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape)
    //     : BaseModel() {
    //     inplanes = 64;
    //
    //     conv1 = torch::nn::Sequential(
    //         torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
    //         torch::nn::BatchNorm2d(64),
    //         torch::nn::ReLU()
    //     );
    //     maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
    //
    //     layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
    //     layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
    //     layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
    //     layer3 = makeLayerFromResidualBlock(512, layers[3], 2);
    //
    //     register_module("conv1", conv1);
    //     register_module("layer0", layer0);
    //     register_module("layer1", layer1);
    //     register_module("layer2", layer2);
    //     register_module("layer3", layer3);
    //
    //     // Compute flattened size dynamically
    //     torch::NoGradGuard no_grad;
    //     auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
    //     auto x = conv1->forward(dummy_input);
    //     x = maxpool->forward(x);
    //     x = layer0->forward(x);
    //     x = layer1->forward(x);
    //     x = layer2->forward(x);
    //     x = layer3->forward(x);
    //
    //     // Adaptive pooling instead of fixed 7x7 pooling
    //     auto spatial_dims = x.sizes().slice(2); // Get H and W after all convs
    //     // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
    //     avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
    //     x = avgpool->forward(x);
    //     int flattened_size = x.numel() / x.size(0);
    //
    //     fc = torch::nn::Linear(flattened_size, num_classes);
    //
    //     register_module("avgpool", avgpool);
    //     register_module("fc", fc);
    // }
    //
    // // torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    // //     torch::nn::Sequential downsample = nullptr;
    // //     if (stride != 1 || inplanes != planes) {
    // //         downsample = torch::nn::Sequential();
    // //         downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)));
    // //         downsample->push_back(torch::nn::BatchNorm2d(planes));
    // //     }
    // //
    // //     auto layers = torch::nn::Sequential();
    // //     layers->push_back(ResidualBlock(inplanes, planes, stride, downsample));
    // //     inplanes = planes;
    // //     for (int i = 1; i < blocks; ++i) {
    // //         layers->push_back(ResidualBlock(inplanes, planes));
    // //     }
    // //     return layers;
    // // }
    //
    //
    // torch::nn::Sequential WideResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    //     torch::nn::Sequential downsample = nullptr;
    //     if (stride != 1 || inplanes != planes) {
    //         downsample = torch::nn::Sequential();
    //         //                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
    //         torch::nn::Conv2d convd = torch::nn::Conv2d(
    //             torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride).padding(0));
    //         downsample->push_back(convd);
    //         //                    nn.BatchNorm2d(planes),
    //         torch::nn::BatchNorm2d batchd = torch::nn::BatchNorm2d(planes);
    //         downsample->push_back(batchd);
    //     }
    //     torch::nn::Sequential layers = torch::nn::Sequential();
    //     ResidualBlock rb = ResidualBlock(inplanes, planes, stride, downsample);
    //     layers->push_back(rb);
    //     inplanes = planes;
    //     for (int i = 1; i < blocks; i++) {
    //         ResidualBlock rbt = ResidualBlock(inplanes, planes);
    //         layers->push_back(rbt);
    //     }
    //     return layers;
    // }
    //
    // torch::Tensor WideResNet::forward(torch::Tensor x) const {
    //     x = conv1->forward(x);
    //     x = maxpool->forward(x);
    //     x = layer0->forward(x);
    //     x = layer1->forward(x);
    //     x = layer2->forward(x);
    //     x = layer3->forward(x);
    //     x = avgpool->forward(x);
    //     x = x.view({x.size(0), -1});
    //     x = fc->forward(x);
    //     return x;
    // }
}
