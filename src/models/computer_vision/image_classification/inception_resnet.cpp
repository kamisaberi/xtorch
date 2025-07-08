#include "include/models/computer_vision/image_classification/inception_resnet.h"


using namespace std;


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // --- MODEL DEFINITION ---
// // This section defines the entire InceptionResNetV1 architecture from scratch.
// // Module and submodule names are kept consistent with standard implementations.
//
// struct BasicConv2dImpl : torch::nn::Module {
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::BatchNorm2d bn{nullptr};
//     bool use_relu;
//
//     BasicConv2dImpl(int in_planes, int out_planes, int kernel_size, int stride, int padding, bool relu = true)
//         : use_relu(relu) {
//         conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size)
//                                           .stride(stride).padding(padding).bias(false)));
//         bn = register_module("bn", torch::nn::BatchNorm2d(out_planes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = conv(x);
//         x = bn(x);
//         if (use_relu) {
//             x = torch::relu(x);
//         }
//         return x;
//     }
// };
// TORCH_MODULE(BasicConv2d);
//
// // Blocks from InceptionResNetV1, similar to the previous example
// struct InceptionResNetAImpl : torch::nn::Module { /* ... same as previous answer ... */ };
// TORCH_MODULE(InceptionResNetA);
// struct InceptionResNetBImpl : torch::nn::Module { /* ... same as previous answer ... */ };
// TORCH_MODULE(InceptionResNetB);
// struct InceptionResNetCImpl : torch::nn::Module { /* ... same as previous answer ... */ };
// TORCH_MODULE(InceptionResNetC);
// struct ReductionAImpl : torch::nn::Module { /* ... same as previous answer ... */ };
// TORCH_MODULE(ReductionA);
// struct ReductionBImpl : torch::nn::Module { /* ... same as previous answer ... */ };
// TORCH_MODULE(ReductionB);
//
// // --- NOTE: The block definitions are omitted here for brevity ---
// // --- Please copy the full block definitions from the previous "from-scratch" answer ---
// // --- I will re-include them below for completeness ---
//
// // Inception-ResNet-A block
// struct InceptionResNetAImpl : torch::nn::Module {
//     BasicConv2d b0, b1_0, b1_1, b2_0, b2_1, b2_2;
//     torch::nn::Conv2d conv; double scale;
//     InceptionResNetAImpl(int in_planes, double scale = 1.0) : b0(in_planes, 32, 1, 1, 0), b1_0(in_planes, 32, 1, 1, 0), b1_1(32, 32, 3, 1, 1), b2_0(in_planes, 32, 1, 1, 0), b2_1(32, 48, 3, 1, 1), b2_2(48, 64, 3, 1, 1), conv(torch::nn::Conv2dOptions(128, 384, 1).stride(1).padding(0)), scale(scale) {
//         register_module("b0", b0); register_module("b1_0", b1_0); register_module("b1_1", b1_1); register_module("b2_0", b2_0); register_module("b2_1", b2_1); register_module("b2_2", b2_2); register_module("conv2d", conv); }
//     torch::Tensor forward(torch::Tensor x) { auto x0 = b0->forward(x); auto x1 = b1_1->forward(b1_0->forward(x)); auto x2 = b2_2->forward(b2_1->forward(b2_0->forward(x))); auto mixed = torch::cat({x0, x1, x2}, 1); auto up = conv(mixed); x = x + up * scale; return torch::relu(x); }
// };
// TORCH_MODULE(InceptionResNetA);
//
// // Inception-ResNet-B block
// struct InceptionResNetBImpl : torch::nn::Module {
//     BasicConv2d b0, b1_0, b1_1, b1_2; torch::nn::Conv2d conv; double scale;
//     InceptionResNetBImpl(int in_planes, double scale = 1.0) : b0(in_planes, 192, 1, 1, 0), b1_0(in_planes, 128, 1, 1, 0), b1_1(128, 160, {1, 7}, 1, {0, 3}), b1_2(160, 192, {7, 1}, 1, {3, 0}), conv(torch::nn::Conv2dOptions(384, 896, 1).stride(1).padding(0)), scale(scale) {
//         register_module("b0", b0); register_module("b1_0", b1_0); register_module("b1_1", b1_1); register_module("b1_2", b1_2); register_module("conv2d", conv); }
//     torch::Tensor forward(torch::Tensor x) { auto x0 = b0->forward(x); auto x1 = b1_2->forward(b1_1->forward(b1_0->forward(x))); auto mixed = torch::cat({x0, x1}, 1); auto up = conv(mixed); x = x + up * scale; return torch::relu(x); }
// };
// TORCH_MODULE(InceptionResNetB);
//
// // Inception-ResNet-C block
// struct InceptionResNetCImpl : torch::nn::Module {
//     BasicConv2d b0, b1_0, b1_1, b1_2; torch::nn::Conv2d conv; double scale;
//     InceptionResNetCImpl(int in_planes, double scale = 1.0) : b0(in_planes, 192, 1, 1, 0), b1_0(in_planes, 192, 1, 1, 0), b1_1(192, 224, {1, 3}, 1, {0, 1}), b1_2(224, 256, {3, 1}, 1, {1, 0}), conv(torch::nn::Conv2dOptions(448, 1792, 1).stride(1).padding(0)), scale(scale) {
//         register_module("b0", b0); register_module("b1_0", b1_0); register_module("b1_1", b1_1); register_module("b1_2", b1_2); register_module("conv2d", conv); }
//     torch::Tensor forward(torch::Tensor x) { auto x0 = b0->forward(x); auto x1 = b1_2->forward(b1_1->forward(b1_0->forward(x))); auto mixed = torch::cat({x0, x1}, 1); auto up = conv(mixed); x = x + up * scale; return torch::relu(x); }
// };
// TORCH_MODULE(InceptionResNetC);
//
// // Reduction-A block
// struct ReductionAImpl : torch::nn::Module {
//     BasicConv2d b0, b1_0, b1_1, b1_2; torch::nn::MaxPool2d b2;
//     ReductionAImpl(int in_planes, int k, int l, int m, int n) : b0(in_planes, n, 3, 2, 0), b1_0(in_planes, k, 1, 1, 0), b1_1(k, l, 3, 1, 1), b1_2(l, m, 3, 2, 0), b2(torch::nn::MaxPool2dOptions(3).stride(2)) {
//         register_module("b0", b0); register_module("b1_0", b1_0); register_module("b1_1", b1_1); register_module("b1_2", b1_2); register_module("b2", b2); }
//     torch::Tensor forward(torch::Tensor x) { auto x0 = b0->forward(x); auto x1 = b1_2->forward(b1_1->forward(b1_0->forward(x))); auto x2 = b2->forward(x); return torch::cat({x0, x1, x2}, 1); }
// };
// TORCH_MODULE(ReductionA);
//
// // Reduction-B block
// struct ReductionBImpl : torch::nn::Module {
//     BasicConv2d b0_0, b0_1, b1_0, b1_1, b2_0, b2_1, b2_2; torch::nn::MaxPool2d b3;
//     ReductionBImpl(int in_planes) : b0_0(in_planes, 256, 1, 1, 0), b0_1(256, 384, 3, 2, 0), b1_0(in_planes, 256, 1, 1, 0), b1_1(256, 288, 3, 2, 0), b2_0(in_planes, 256, 1, 1, 0), b2_1(256, 288, 3, 1, 1), b2_2(288, 320, 3, 2, 0), b3(torch::nn::MaxPool2dOptions(3).stride(2)) {
//         register_module("b0_0", b0_0); register_module("b0_1", b0_1); register_module("b1_0", b1_0); register_module("b1_1", b1_1); register_module("b2_0", b2_0); register_module("b2_1", b2_1); register_module("b2_2", b2_2); register_module("b3", b3); }
//     torch::Tensor forward(torch::Tensor x) { auto x0 = b0_1->forward(b0_0->forward(x)); auto x1 = b1_1->forward(b1_0->forward(x)); auto x2 = b2_2->forward(b2_1->forward(b2_0->forward(x))); auto x3 = b3->forward(x); return torch::cat({x0, x1, x2, x3}, 1); }
// };
// TORCH_MODULE(ReductionB);
//
//
// // The complete InceptionResNetV1 model, adapted for MNIST
// struct InceptionResNetV1Impl : torch::nn::Module {
//     BasicConv2d conv2d_1a, conv2d_2a, conv2d_2b;
//     torch::nn::MaxPool2d maxpool_3a;
//     BasicConv2d conv2d_3b, conv2d_4a;
//     torch::nn::MaxPool2d maxpool_5a;
//     torch::nn::Sequential repeat, repeat_1, repeat_2;
//     ReductionA mixed_6a;
//     ReductionB mixed_7a;
//     BasicConv2d block8;
//     torch::nn::AdaptiveAvgPool2d avgpool_1a;
//     torch::nn::Dropout dropout;
//     torch::nn::Linear logits;
//
//     InceptionResNetV1Impl(int num_classes = 10)
//         // **MODIFICATION 1: Input channels changed from 3 to 1 for MNIST**
//         : conv2d_1a(1, 32, 3, 2, 0),
//           conv2d_2a(32, 32, 3, 1, 0),
//           conv2d_2b(32, 64, 3, 1, 1),
//           maxpool_3a(torch::nn::MaxPool2dOptions(3).stride(2)),
//           conv2d_3b(64, 80, 1, 1, 0),
//           conv2d_4a(80, 192, 3, 1, 0),
//           maxpool_5a(torch::nn::MaxPool2dOptions(3).stride(2)),
//           mixed_6a(192, 256, 256, 256, 384),
//           mixed_7a(896),
//           block8(1792, 1792, 1, 1, 0, false),
//           avgpool_1a(torch::nn::AdaptiveAvgPool2dOptions(1)),
//           dropout(0.6),
//           // **MODIFICATION 2: Output classes changed to 10 for MNIST**
//           logits(1792, num_classes)
//     {
//         // Register modules in the correct order with correct names
//         register_module("conv2d_1a", conv2d_1a);
//         register_module("conv2d_2a", conv2d_2a);
//         register_module("conv2d_2b", conv2d_2b);
//         register_module("maxpool_3a", maxpool_3a);
//         register_module("conv2d_3b", conv2d_3b);
//         register_module("conv2d_4a", conv2d_4a);
//         register_module("maxpool_5a", maxpool_5a);
//
//         repeat = torch::nn::Sequential();
//         for (int i = 0; i < 5; ++i) repeat->push_back("block" + std::to_string(i), InceptionResNetA(384, 0.17));
//         register_module("repeat", repeat);
//
//         register_module("mixed_6a", mixed_6a);
//
//         repeat_1 = torch::nn::Sequential();
//         for (int i = 0; i < 10; ++i) repeat_1->push_back("block" + std::to_string(i), InceptionResNetB(896, 0.10));
//         register_module("repeat_1", repeat_1);
//
//         register_module("mixed_7a", mixed_7a);
//
//         repeat_2 = torch::nn::Sequential();
//         for (int i = 0; i < 5; ++i) repeat_2->push_back("block" + std::to_string(i), InceptionResNetC(1792, 0.20));
//         register_module("repeat_2", repeat_2);
//
//         register_module("block8", block8);
//         register_module("avgpool_1a", avgpool_1a);
//         register_module("dropout", dropout);
//         register_module("logits", logits);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = conv2d_1a->forward(x); x = conv2d_2a->forward(x); x = conv2d_2b->forward(x);
//         x = maxpool_3a->forward(x); x = conv2d_3b->forward(x); x = conv2d_4a->forward(x);
//         x = maxpool_5a->forward(x);
//         x = repeat->forward(x); x = mixed_6a->forward(x); x = repeat_1->forward(x);
//         x = mixed_7a->forward(x); x = repeat_2->forward(x);
//         x = block8->forward(x); x = avgpool_1a->forward(x);
//         x = dropout->forward(x);
//         x = x.view({x.size(0), -1});
//         x = logits->forward(x);
//         // **MODIFICATION 3: No L2 normalization. Return raw logits for CrossEntropyLoss.**
//         return x;
//     }
// };
// TORCH_MODULE(InceptionResNetV1);
//
//
// // --- Training and Evaluation Logic ---
//
// template <typename DataLoader>
// void train(
//     InceptionResNetV1& model,
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
//         AT_ASSERT(!std::isnan(loss.template item<float>()));
//         loss.backward();
//         optimizer.step();
//
//         if (batch_idx++ % 20 == 0) {
//             std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
//                 epoch, batch_idx * batch.data.size(0),
//                 dataset_size, loss.template item<float>());
//         }
//     }
// }
//
// template <typename DataLoader>
// void test(
//     InceptionResNetV1& model,
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
//         test_loss += torch::nn::functional::cross_entropy(output, targets,
//             /*weight=*/{}, torch::Reduction::Sum).template item<double>();
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
// // --- Main Function ---
//
// int main() {
//     torch::manual_seed(1);
//
//     // --- Hyperparameters ---
//     const int64_t kTrainBatchSize = 64;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 10;
//     const double kLearningRate = 0.001;
//     const int64_t kImageSize = 96; // Upsample MNIST images to this size
//
//     // --- Device Setup ---
//     torch::Device device = torch::kCPU;
//     if (torch::cuda::is_available()) {
//         std::cout << "CUDA is available! Training on GPU." << std::endl;
//         device = torch::kCUDA;
//     }
//
//     // --- Model and Optimizer ---
//     InceptionResNetV1 model(10); // 10 classes for MNIST
//     model->to(device);
//
//     torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(kLearningRate));
//
//     // --- Data Loading ---
//     // InceptionResNetV1 is too large for 28x28, so we resize the images.
//     auto train_dataset = torch::data::datasets::MNIST("./mnist_data")
//         .map(torch::data::transforms::Resize<>({kImageSize, kImageSize}))
//         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//         .map(torch::data::transforms::Stack<>());
//
//     const size_t train_dataset_size = train_dataset.size().value();
//     auto train_loader = torch::data::make_data_loader(
//         std::move(train_dataset), kTrainBatchSize);
//
//     auto test_dataset = torch::data::datasets::MNIST("./mnist_data", torch::data::datasets::MNIST::Mode::kTest)
//         .map(torch::data::transforms::Resize<>({kImageSize, kImageSize}))
//         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//         .map(torch::data::transforms::Stack<>());
//
//     const size_t test_dataset_size = test_dataset.size().value();
//     auto test_loader = torch::data::make_data_loader(
//         std::move(test_dataset), kTestBatchSize);
//
//     std::cout << "Training InceptionResNetV1 on MNIST..." << std::endl;
//     std::cout << "NOTE: This model is very large for MNIST and training will be slow on CPU." << std::endl;
//
//     // --- Start Training ---
//     for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
//         train(model, *train_loader, optimizer, epoch, train_dataset_size, device);
//         test(model, *test_loader, test_dataset_size, device);
//     }
//
//     std::cout << "Training finished." << std::endl;
//
//     return 0;
// }


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // --- MODEL BUILDING BLOCKS (FROM SCRATCH) ---
//
// // A standard Conv->BN->ReLU block used throughout the network.
// struct BasicConv2dImpl : torch::nn::Module {
//     torch::nn::Conv2d conv{nullptr};
//     torch::nn::BatchNorm2d bn{nullptr};
//
//     BasicConv2dImpl(int in_channels, int out_channels, torch::ExpandingArray<2> kernel_size,
//                     torch::ExpandingArray<2> stride = 1, torch::ExpandingArray<2> padding = 0) {
//         conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
//                                           .stride(stride).padding(padding).bias(false)));
//         bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = conv(x);
//         x = bn(x);
//         return torch::relu(x);
//     }
// };
// TORCH_MODULE(BasicConv2d);
//
// // The complex stem of the InceptionResNetV2.
// struct StemImpl : torch::nn::Module {
//     BasicConv2d conv2d_1a, conv2d_2a, conv2d_2b;
//     torch::nn::MaxPool2d maxpool_3a;
//     BasicConv2d conv2d_3b, conv2d_4a;
//     BasicConv2d branch_0_conv, branch_1_conv_1, branch_1_conv_2;
//     torch::nn::MaxPool2d branch_pool;
//
//     StemImpl(int in_channels)
//         : conv2d_1a(in_channels, 32, 3, 2, 0),
//           conv2d_2a(32, 32, 3, 1, 0),
//           conv2d_2b(32, 64, 3, 1, 1),
//           maxpool_3a(torch::nn::MaxPool2dOptions(3).stride(2)),
//           conv2d_3b(64, 80, 1, 1, 0),
//           conv2d_4a(80, 192, 3, 1, 0),
//           branch_0_conv(192, 96, 3, 2, 0),
//           branch_1_conv_1(192, 64, 1, 1, 0),
//           branch_1_conv_2(64, 96, 3, 1, 1),
//           branch_pool(torch::nn::MaxPool2dOptions(3).stride(2))
//     {
//         register_module("conv2d_1a", conv2d_1a);
//         register_module("conv2d_2a", conv2d_2a);
//         register_module("conv2d_2b", conv2d_2b);
//         register_module("maxpool_3a", maxpool_3a);
//         register_module("conv2d_3b", conv2d_3b);
//         register_module("conv2d_4a", conv2d_4a);
//         register_module("branch_0_conv", branch_0_conv);
//         register_module("branch_1_conv_1", branch_1_conv_1);
//         register_module("branch_1_conv_2", branch_1_conv_2);
//         register_module("branch_pool", branch_pool);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = conv2d_1a(x); x = conv2d_2a(x); x = conv2d_2b(x);
//         x = maxpool_3a(x);
//         x = conv2d_3b(x); x = conv2d_4a(x);
//
//         auto branch_0 = branch_0_conv(x);
//         auto branch_1 = branch_1_conv_1(x);
//         branch_1 = branch_1_conv_2(branch_1);
//         auto branch_pool_out = branch_pool(x);
//
//         x = torch::cat({branch_0, branch_1, branch_pool_out}, 1);
//         return x;
//     }
// };
// TORCH_MODULE(Stem);
//
// // Inception-ResNet-A Block for V2
// struct InceptionResNetAImpl : torch::nn::Module {
//     BasicConv2d b1, b2_1, b2_2, b3_1, b3_2, b3_3;
//     torch::nn::Conv2d conv; double scale;
//     InceptionResNetAImpl(int in_channels, double scale = 1.0)
//         : b1(in_channels, 32, 1, 1, 0),
//           b2_1(in_channels, 32, 1, 1, 0),
//           b2_2(32, 32, 3, 1, 1),
//           b3_1(in_channels, 32, 1, 1, 0),
//           b3_2(32, 48, 3, 1, 1),
//           b3_3(48, 64, 3, 1, 1),
//           conv(torch::nn::Conv2dOptions(128, 320, 1).bias(true)),
//           scale(scale)
//     {
//         register_module("b1", b1); register_module("b2_1", b2_1); register_module("b2_2", b2_2);
//         register_module("b3_1", b3_1); register_module("b3_2", b3_2); register_module("b3_3", b3_3);
//         register_module("conv", conv);
//     }
//     torch::Tensor forward(torch::Tensor x) {
//         auto x_b1 = b1(x);
//         auto x_b2 = b2_2(b2_1(x));
//         auto x_b3 = b3_3(b3_2(b3_1(x)));
//         auto mixed = torch::cat({x_b1, x_b2, x_b3}, 1);
//         auto up = torch::relu(conv(mixed));
//         x = x + up * scale;
//         return torch::relu(x);
//     }
// };
// TORCH_MODULE(InceptionResNetA);
//
// // Inception-ResNet-B Block for V2
// struct InceptionResNetBImpl : torch::nn::Module {
//     BasicConv2d b1, b2_1, b2_2, b2_3;
//     torch::nn::Conv2d conv; double scale;
//     InceptionResNetBImpl(int in_channels, double scale = 1.0)
//         : b1(in_channels, 192, 1, 1, 0),
//           b2_1(in_channels, 128, 1, 1, 0),
//           b2_2(128, 160, {1, 7}, 1, {0, 3}),
//           b2_3(160, 192, {7, 1}, 1, {3, 0}),
//           conv(torch::nn::Conv2dOptions(384, 1088, 1).bias(true)),
//           scale(scale)
//     {
//         register_module("b1", b1); register_module("b2_1", b2_1); register_module("b2_2", b2_2);
//         register_module("b2_3", b2_3); register_module("conv", conv);
//     }
//     torch::Tensor forward(torch::Tensor x) {
//         auto x_b1 = b1(x);
//         auto x_b2 = b2_3(b2_2(b2_1(x)));
//         auto mixed = torch::cat({x_b1, x_b2}, 1);
//         auto up = torch::relu(conv(mixed));
//         x = x + up * scale;
//         return torch::relu(x);
//     }
// };
// TORCH_MODULE(InceptionResNetB);
//
// // Inception-ResNet-C Block for V2
// struct InceptionResNetCImpl : torch::nn::Module {
//     BasicConv2d b1, b2_1, b2_2, b2_3;
//     torch::nn::Conv2d conv; double scale;
//     InceptionResNetCImpl(int in_channels, double scale = 1.0)
//         : b1(in_channels, 192, 1, 1, 0),
//           b2_1(in_channels, 192, 1, 1, 0),
//           b2_2(192, 224, {1, 3}, 1, {0, 1}),
//           b2_3(224, 256, {3, 1}, 1, {1, 0}),
//           conv(torch::nn::Conv2dOptions(448, 2080, 1).bias(true)),
//           scale(scale)
//     {
//         register_module("b1", b1); register_module("b2_1", b2_1); register_module("b2_2", b2_2);
//         register_module("b2_3", b2_3); register_module("conv", conv);
//     }
//     torch::Tensor forward(torch::Tensor x) {
//         auto x_b1 = b1(x);
//         auto x_b2 = b2_3(b2_2(b2_1(x)));
//         auto mixed = torch::cat({x_b1, x_b2}, 1);
//         auto up = torch::relu(conv(mixed));
//         x = x + up * scale;
//         return torch::relu(x);
//     }
// };
// TORCH_MODULE(InceptionResNetC);
//
// // Reduction-A Block for V2 (downsamples)
// struct ReductionAImpl : torch::nn::Module {
//     torch::nn::MaxPool2d b1;
//     BasicConv2d b2, b3_1, b3_2, b3_3;
//     ReductionAImpl(int in_channels, int k, int l, int m, int n)
//         : b1(torch::nn::MaxPool2dOptions(3).stride(2)),
//           b2(in_channels, n, 3, 2, 0),
//           b3_1(in_channels, k, 1, 1, 0),
//           b3_2(k, l, 3, 1, 1),
//           b3_3(l, m, 3, 2, 0)
//     {
//         register_module("b1", b1); register_module("b2", b2);
//         register_module("b3_1", b3_1); register_module("b3_2", b3_2); register_module("b3_3", b3_3);
//     }
//     torch::Tensor forward(torch::Tensor x) {
//         auto x_b1 = b1(x);
//         auto x_b2 = b2(x);
//         auto x_b3 = b3_3(b3_2(b3_1(x)));
//         return torch::cat({x_b1, x_b2, x_b3}, 1);
//     }
// };
// TORCH_MODULE(ReductionA);
//
// // Reduction-B Block for V2 (downsamples)
// struct ReductionBImpl : torch::nn::Module {
//     torch::nn::MaxPool2d b1;
//     BasicConv2d b2_1, b2_2, b3_1, b3_2, b4_1, b4_2, b4_3;
//     ReductionBImpl(int in_channels)
//         : b1(torch::nn::MaxPool2dOptions(3).stride(2)),
//           b2_1(in_channels, 256, 1, 1, 0),
//           b2_2(256, 384, 3, 2, 0),
//           b3_1(in_channels, 256, 1, 1, 0),
//           b3_2(256, 288, 3, 2, 0),
//           b4_1(in_channels, 256, 1, 1, 0),
//           b4_2(256, 288, 3, 1, 1),
//           b4_3(288, 320, 3, 2, 0)
//     {
//         register_module("b1", b1); register_module("b2_1", b2_1); register_module("b2_2", b2_2);
//         register_module("b3_1", b3_1); register_module("b3_2", b3_2);
//         register_module("b4_1", b4_1); register_module("b4_2", b4_2); register_module("b4_3", b4_3);
//     }
//     torch::Tensor forward(torch::Tensor x) {
//         auto x_b1 = b1(x);
//         auto x_b2 = b2_2(b2_1(x));
//         auto x_b3 = b3_2(b3_1(x));
//         auto x_b4 = b4_3(b4_2(b4_1(x)));
//         return torch::cat({x_b1, x_b2, x_b3, x_b4}, 1);
//     }
// };
// TORCH_MODULE(ReductionB);
//
// // The Full InceptionResNetV2 model adapted for MNIST
// struct InceptionResNetV2Impl : torch::nn::Module {
//     Stem stem;
//     torch::nn::Sequential repeat_a, repeat_b, repeat_c;
//     ReductionA reduction_a;
//     ReductionB reduction_b;
//     torch::nn::AdaptiveAvgPool2d avgpool;
//     torch::nn::Dropout dropout;
//     torch::nn::Linear logits;
//
//     InceptionResNetV2Impl(int num_classes = 10)
//         // **MODIFICATION 1: Input channels changed from 3 to 1 for MNIST**
//         : stem(1),
//           reduction_a(320, 256, 256, 384, 384),
//           reduction_b(1088),
//           avgpool(torch::nn::AdaptiveAvgPool2dOptions(1)),
//           dropout(0.8),
//           // **MODIFICATION 2: Output classes changed to 10 for MNIST**
//           logits(2080, num_classes)
//     {
//         register_module("stem", stem);
//
//         repeat_a = torch::nn::Sequential();
//         for (int i=0; i<5; ++i) repeat_a->push_back(InceptionResNetA(320, 0.17));
//         register_module("repeat_a", repeat_a);
//
//         register_module("reduction_a", reduction_a);
//
//         repeat_b = torch::nn::Sequential();
//         for (int i=0; i<10; ++i) repeat_b->push_back(InceptionResNetB(1088, 0.10));
//         register_module("repeat_b", repeat_b);
//
//         register_module("reduction_b", reduction_b);
//
//         repeat_c = torch::nn::Sequential();
//         for (int i=0; i<5; ++i) repeat_c->push_back(InceptionResNetC(2080, 0.20));
//         register_module("repeat_c", repeat_c);
//
//         register_module("avgpool", avgpool);
//         register_module("dropout", dropout);
//         register_module("logits", logits);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = stem->forward(x);
//         x = repeat_a->forward(x);
//         x = reduction_a->forward(x);
//         x = repeat_b->forward(x);
//         x = reduction_b->forward(x);
//         x = repeat_c->forward(x);
//         x = avgpool->forward(x);
//         x = dropout->forward(x);
//         x = x.view({x.size(0), -1});
//         // **MODIFICATION 3: Return raw logits for CrossEntropyLoss**
//         x = logits->forward(x);
//         return x;
//     }
// };
// TORCH_MODULE(InceptionResNetV2);
//
// // --- TRAINING AND EVALUATION LOGIC (Re-usable) ---
// template <typename DataLoader>
// void train(InceptionResNetV2& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer,
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
//         if (batch_idx++ % 20 == 0) {
//             std::printf("\rTrain Epoch: %ld [%5ld/%5ld] Loss: %.4f",
//                 epoch, batch_idx * batch.data.size(0), dataset_size, loss.template item<float>());
//         }
//     }
// }
//
// template <typename DataLoader>
// void test(InceptionResNetV2& model, DataLoader& data_loader, size_t dataset_size, torch::Device device) {
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
//     // Hyperparameters
//     const int64_t kTrainBatchSize = 64;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 10;
//     const double kLearningRate = 0.001;
//     const int64_t kImageSize = 96; // Upsample MNIST to this size
//
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training on " << device << std::endl;
//
//     // Model and Optimizer
//     InceptionResNetV2 model(10);
//     model->to(device);
//     torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(kLearningRate));
//
//     // Data Loaders with Resize Transform
//     auto train_dataset = torch::data::datasets::MNIST("./mnist_data")
//         .map(torch::data::transforms::Resize<>({kImageSize, kImageSize}))
//         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//         .map(torch::data::transforms::Stack<>());
//     const size_t train_dataset_size = train_dataset.size().value();
//     auto train_loader = torch::data::make_data_loader(std::move(train_dataset), kTrainBatchSize);
//
//     auto test_dataset = torch::data::datasets::MNIST("./mnist_data", torch::data::datasets::MNIST::Mode::kTest)
//         .map(torch::data::transforms::Resize<>({kImageSize, kImageSize}))
//         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
//         .map(torch::data::transforms::Stack<>());
//     const size_t test_dataset_size = test_dataset.size().value();
//     auto test_loader = torch::data::make_data_loader(std::move(test_dataset), kTestBatchSize);
//
//     std::cout << "Training InceptionResNetV2 on MNIST..." << std::endl;
//     std::cout << "!!! WARNING: This model is EXTREMELY large. Training on CPU will be impractically slow. A GPU is strongly recommended. !!!" << std::endl;
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
    BasicConv2dImpl::BasicConv2dImpl(int in_planes, int out_planes, int kernel_size, int stride, int padding, bool relu)
        : use_relu(relu)
    {
        conv = register_module("conv", torch::nn::Conv2d(
                                   torch::nn::Conv2dOptions(in_planes, out_planes, kernel_size)
                                   .stride(stride).padding(padding).bias(false)));
        bn = register_module("bn", torch::nn::BatchNorm2d(out_planes));
    }

    torch::Tensor BasicConv2dImpl::forward(torch::Tensor x)
    {
        x = conv(x);
        x = bn(x);
        if (use_relu)
        {
            x = torch::relu(x);
        }
        return x;
    }


    InceptionResNetAImpl::InceptionResNetAImpl(int in_planes, double scale = 1.0) : b0(in_planes, 32, 1, 1, 0),
        b1_0(in_planes, 32, 1, 1, 0), b1_1(32, 32, 3, 1, 1),
        b2_0(in_planes, 32, 1, 1, 0), b2_1(32, 48, 3, 1, 1),
        b2_2(48, 64, 3, 1, 1),
        conv(torch::nn::Conv2dOptions(128, 384, 1).stride(1).
                                                   padding(0)), scale(scale)
    {
        register_module("b0", b0);
        register_module("b1_0", b1_0);
        register_module("b1_1", b1_1);
        register_module("b2_0", b2_0);
        register_module("b2_1", b2_1);
        register_module("b2_2", b2_2);
        register_module("conv2d", conv);
    }

    torch::Tensor InceptionResNetAImpl::forward(torch::Tensor x)
    {
        auto x0 = b0->forward(x);
        auto x1 = b1_1->forward(b1_0->forward(x));
        auto x2 = b2_2->forward(b2_1->forward(b2_0->forward(x)));
        auto mixed = torch::cat({x0, x1, x2}, 1);
        auto up = conv(mixed);
        x = x + up * scale;
        return torch::relu(x);
    }

    InceptionResNetBImpl::InceptionResNetBImpl(int in_planes, double scale = 1.0) : b0(in_planes, 192, 1, 1, 0),
        b1_0(in_planes, 128, 1, 1, 0),
        b1_1(128, 160, {1, 7}, 1, {0, 3}),
        b1_2(160, 192, {7, 1}, 1, {3, 0}),
        conv(torch::nn::Conv2dOptions(384, 896, 1).stride(1).
                                                   padding(0)), scale(scale)
    {
        register_module("b0", b0);
        register_module("b1_0", b1_0);
        register_module("b1_1", b1_1);
        register_module("b1_2", b1_2);
        register_module("conv2d", conv);
    }

    torch::Tensor InceptionResNetBImpl::forward(torch::Tensor x)
    {
        auto x0 = b0->forward(x);
        auto x1 = b1_2->forward(b1_1->forward(b1_0->forward(x)));
        auto mixed = torch::cat({x0, x1}, 1);
        auto up = conv(mixed);
        x = x + up * scale;
        return torch::relu(x);
    }

    InceptionResNetCImpl::InceptionResNetCImpl(int in_planes, double scale = 1.0) : b0(in_planes, 192, 1, 1, 0),
        b1_0(in_planes, 192, 1, 1, 0),
        b1_1(192, 224, {1, 3}, 1, {0, 1}),
        b1_2(224, 256, {3, 1}, 1, {1, 0}),
        conv(torch::nn::Conv2dOptions(448, 1792, 1).stride(1).
                                                    padding(0)), scale(scale)
    {
        register_module("b0", b0);
        register_module("b1_0", b1_0);
        register_module("b1_1", b1_1);
        register_module("b1_2", b1_2);
        register_module("conv2d", conv);
    }

    torch::Tensor InceptionResNetCImpl::forward(torch::Tensor x)
    {
        auto x0 = b0->forward(x);
        auto x1 = b1_2->forward(b1_1->forward(b1_0->forward(x)));
        auto mixed = torch::cat({x0, x1}, 1);
        auto up = conv(mixed);
        x = x + up * scale;
        return torch::relu(x);
    }

    ReductionAImpl::ReductionAImpl(int in_planes, int k, int l, int m, int n) : b0(in_planes, n, 3, 2, 0),
        b1_0(in_planes, k, 1, 1, 0), b1_1(k, l, 3, 1, 1),
        b1_2(l, m, 3, 2, 0),
        b2(torch::nn::MaxPool2dOptions(3).stride(2))
    {
        register_module("b0", b0);
        register_module("b1_0", b1_0);
        register_module("b1_1", b1_1);
        register_module("b1_2", b1_2);
        register_module("b2", b2);
    }

    torch::Tensor ReductionAImpl::forward(torch::Tensor x)
    {
        auto x0 = b0->forward(x);
        auto x1 = b1_2->forward(b1_1->forward(b1_0->forward(x)));
        auto x2 = b2->forward(x);
        return torch::cat({x0, x1, x2}, 1);
    }

    // Reduction-B block
    struct ReductionBImpl : torch::nn::Module
    {
        BasicConv2d b0_0, b0_1, b1_0, b1_1, b2_0, b2_1, b2_2;
        torch::nn::MaxPool2d b3;

        ReductionBImpl(int in_planes) : b0_0(in_planes, 256, 1, 1, 0), b0_1(256, 384, 3, 2, 0),
                                        b1_0(in_planes, 256, 1, 1, 0), b1_1(256, 288, 3, 2, 0),
                                        b2_0(in_planes, 256, 1, 1, 0), b2_1(256, 288, 3, 1, 1), b2_2(288, 320, 3, 2, 0),
                                        b3(torch::nn::MaxPool2dOptions(3).stride(2))
        {
            register_module("b0_0", b0_0);
            register_module("b0_1", b0_1);
            register_module("b1_0", b1_0);
            register_module("b1_1", b1_1);
            register_module("b2_0", b2_0);
            register_module("b2_1", b2_1);
            register_module("b2_2", b2_2);
            register_module("b3", b3);
        }

        torch::Tensor forward(torch::Tensor x)
        {
            auto x0 = b0_1->forward(b0_0->forward(x));
            auto x1 = b1_1->forward(b1_0->forward(x));
            auto x2 = b2_2->forward(b2_1->forward(b2_0->forward(x)));
            auto x3 = b3->forward(x);
            return torch::cat({x0, x1, x2, x3}, 1);
        }
    };

    TORCH_MODULE(ReductionB);


    // The complex stem of the InceptionResNetV2.
    struct StemImpl : torch::nn::Module
    {
        BasicConv2d conv2d_1a, conv2d_2a, conv2d_2b;
        torch::nn::MaxPool2d maxpool_3a;
        BasicConv2d conv2d_3b, conv2d_4a;
        BasicConv2d branch_0_conv, branch_1_conv_1, branch_1_conv_2;
        torch::nn::MaxPool2d branch_pool;

        StemImpl(int in_channels)
            : conv2d_1a(in_channels, 32, 3, 2, 0),
              conv2d_2a(32, 32, 3, 1, 0),
              conv2d_2b(32, 64, 3, 1, 1),
              maxpool_3a(torch::nn::MaxPool2dOptions(3).stride(2)),
              conv2d_3b(64, 80, 1, 1, 0),
              conv2d_4a(80, 192, 3, 1, 0),
              branch_0_conv(192, 96, 3, 2, 0),
              branch_1_conv_1(192, 64, 1, 1, 0),
              branch_1_conv_2(64, 96, 3, 1, 1),
              branch_pool(torch::nn::MaxPool2dOptions(3).stride(2))
        {
            register_module("conv2d_1a", conv2d_1a);
            register_module("conv2d_2a", conv2d_2a);
            register_module("conv2d_2b", conv2d_2b);
            register_module("maxpool_3a", maxpool_3a);
            register_module("conv2d_3b", conv2d_3b);
            register_module("conv2d_4a", conv2d_4a);
            register_module("branch_0_conv", branch_0_conv);
            register_module("branch_1_conv_1", branch_1_conv_1);
            register_module("branch_1_conv_2", branch_1_conv_2);
            register_module("branch_pool", branch_pool);
        }

        torch::Tensor forward(torch::Tensor x)
        {
            x = conv2d_1a(x);
            x = conv2d_2a(x);
            x = conv2d_2b(x);
            x = maxpool_3a(x);
            x = conv2d_3b(x);
            x = conv2d_4a(x);

            auto branch_0 = branch_0_conv(x);
            auto branch_1 = branch_1_conv_1(x);
            branch_1 = branch_1_conv_2(branch_1);
            auto branch_pool_out = branch_pool(x);

            x = torch::cat({branch_0, branch_1, branch_pool_out}, 1);
            return x;
        }
    };

    TORCH_MODULE(Stem);


    // The complete InceptionResNetV1 model, adapted for MNIST
    struct InceptionResNetV1Impl : torch::nn::Module
    {
        BasicConv2d conv2d_1a, conv2d_2a, conv2d_2b;
        torch::nn::MaxPool2d maxpool_3a;
        BasicConv2d conv2d_3b, conv2d_4a;
        torch::nn::MaxPool2d maxpool_5a;
        torch::nn::Sequential repeat, repeat_1, repeat_2;
        ReductionA mixed_6a;
        ReductionB mixed_7a;
        BasicConv2d block8;
        torch::nn::AdaptiveAvgPool2d avgpool_1a;
        torch::nn::Dropout dropout;
        torch::nn::Linear logits;

        InceptionResNetV1Impl(int num_classes = 10)
        // **MODIFICATION 1: Input channels changed from 3 to 1 for MNIST**
            : conv2d_1a(1, 32, 3, 2, 0),
              conv2d_2a(32, 32, 3, 1, 0),
              conv2d_2b(32, 64, 3, 1, 1),
              maxpool_3a(torch::nn::MaxPool2dOptions(3).stride(2)),
              conv2d_3b(64, 80, 1, 1, 0),
              conv2d_4a(80, 192, 3, 1, 0),
              maxpool_5a(torch::nn::MaxPool2dOptions(3).stride(2)),
              mixed_6a(192, 256, 256, 256, 384),
              mixed_7a(896),
              block8(1792, 1792, 1, 1, 0, false),
              avgpool_1a(torch::nn::AdaptiveAvgPool2dOptions(1)),
              dropout(0.6),
              // **MODIFICATION 2: Output classes changed to 10 for MNIST**
              logits(1792, num_classes)
        {
            // Register modules in the correct order with correct names
            register_module("conv2d_1a", conv2d_1a);
            register_module("conv2d_2a", conv2d_2a);
            register_module("conv2d_2b", conv2d_2b);
            register_module("maxpool_3a", maxpool_3a);
            register_module("conv2d_3b", conv2d_3b);
            register_module("conv2d_4a", conv2d_4a);
            register_module("maxpool_5a", maxpool_5a);

            repeat = torch::nn::Sequential();
            for (int i = 0; i < 5; ++i) repeat->push_back("block" + std::to_string(i), InceptionResNetA(384, 0.17));
            register_module("repeat", repeat);

            register_module("mixed_6a", mixed_6a);

            repeat_1 = torch::nn::Sequential();
            for (int i = 0; i < 10; ++i) repeat_1->push_back("block" + std::to_string(i), InceptionResNetB(896, 0.10));
            register_module("repeat_1", repeat_1);

            register_module("mixed_7a", mixed_7a);

            repeat_2 = torch::nn::Sequential();
            for (int i = 0; i < 5; ++i) repeat_2->push_back("block" + std::to_string(i), InceptionResNetC(1792, 0.20));
            register_module("repeat_2", repeat_2);

            register_module("block8", block8);
            register_module("avgpool_1a", avgpool_1a);
            register_module("dropout", dropout);
            register_module("logits", logits);
        }

        torch::Tensor forward(torch::Tensor x)
        {
            x = conv2d_1a->forward(x);
            x = conv2d_2a->forward(x);
            x = conv2d_2b->forward(x);
            x = maxpool_3a->forward(x);
            x = conv2d_3b->forward(x);
            x = conv2d_4a->forward(x);
            x = maxpool_5a->forward(x);
            x = repeat->forward(x);
            x = mixed_6a->forward(x);
            x = repeat_1->forward(x);
            x = mixed_7a->forward(x);
            x = repeat_2->forward(x);
            x = block8->forward(x);
            x = avgpool_1a->forward(x);
            x = dropout->forward(x);
            x = x.view({x.size(0), -1});
            x = logits->forward(x);
            // **MODIFICATION 3: No L2 normalization. Return raw logits for CrossEntropyLoss.**
            return x;
        }
    };

    TORCH_MODULE(InceptionResNetV1);


    // The Full InceptionResNetV2 model adapted for MNIST
    struct InceptionResNetV2Impl : torch::nn::Module
    {
        Stem stem;
        torch::nn::Sequential repeat_a, repeat_b, repeat_c;
        ReductionA reduction_a;
        ReductionB reduction_b;
        torch::nn::AdaptiveAvgPool2d avgpool;
        torch::nn::Dropout dropout;
        torch::nn::Linear logits;

        InceptionResNetV2Impl(int num_classes = 10)
        // **MODIFICATION 1: Input channels changed from 3 to 1 for MNIST**
            : stem(1),
              reduction_a(320, 256, 256, 384, 384),
              reduction_b(1088),
              avgpool(torch::nn::AdaptiveAvgPool2dOptions(1)),
              dropout(0.8),
              // **MODIFICATION 2: Output classes changed to 10 for MNIST**
              logits(2080, num_classes)
        {
            register_module("stem", stem);

            repeat_a = torch::nn::Sequential();
            for (int i = 0; i < 5; ++i) repeat_a->push_back(InceptionResNetA(320, 0.17));
            register_module("repeat_a", repeat_a);

            register_module("reduction_a", reduction_a);

            repeat_b = torch::nn::Sequential();
            for (int i = 0; i < 10; ++i) repeat_b->push_back(InceptionResNetB(1088, 0.10));
            register_module("repeat_b", repeat_b);

            register_module("reduction_b", reduction_b);

            repeat_c = torch::nn::Sequential();
            for (int i = 0; i < 5; ++i) repeat_c->push_back(InceptionResNetC(2080, 0.20));
            register_module("repeat_c", repeat_c);

            register_module("avgpool", avgpool);
            register_module("dropout", dropout);
            register_module("logits", logits);
        }

        torch::Tensor forward(torch::Tensor x)
        {
            x = stem->forward(x);
            x = repeat_a->forward(x);
            x = reduction_a->forward(x);
            x = repeat_b->forward(x);
            x = reduction_b->forward(x);
            x = repeat_c->forward(x);
            x = avgpool->forward(x);
            x = dropout->forward(x);
            x = x.view({x.size(0), -1});
            // **MODIFICATION 3: Return raw logits for CrossEntropyLoss**
            x = logits->forward(x);
            return x;
        }
    };

    TORCH_MODULE(InceptionResNetV2);


    // InceptionResNetV1::InceptionResNetV1(int num_classes, int in_channels)
    // {
    // }
    //
    // InceptionResNetV1::InceptionResNetV1(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    // {
    // }
    //
    // void InceptionResNetV1::reset()
    // {
    // }
    //
    // auto InceptionResNetV1::forward(std::initializer_list<std::any> tensors) -> std::any
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
    //
    //     return x;
    // }
    //
    //
    // InceptionResNetV2::InceptionResNetV2(int num_classes, int in_channels)
    // {
    // }
    //
    // InceptionResNetV2::InceptionResNetV2(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    // {
    // }
    //
    // void InceptionResNetV2::reset()
    // {
    // }
    //
    // auto InceptionResNetV2::forward(std::initializer_list<std::any> tensors) -> std::any
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
    //
    //     return x;
    // }
}
