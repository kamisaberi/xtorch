#include "include/models/computer_vision/image_classification/xception.h"


using namespace std;


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // --- The Core Building Block: Depthwise Separable Convolution ---
// struct SeparableConv2d : torch::nn::Module {
//     torch::nn::Conv2d depthwise, pointwise;
//
//     SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
//         // Depthwise convolution: processes each channel independently
//         : depthwise(torch::nn::Conv2dOptions(in_channels, in_channels, kernel_size)
//                         .stride(stride).padding(padding).groups(in_channels).bias(false)),
//           // Pointwise convolution: 1x1 conv to mix channel information
//           pointwise(torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false))
//     {
//         register_module("depthwise", depthwise);
//         register_module("pointwise", pointwise);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         return pointwise(depthwise(x));
//     }
// };
// TORCH_MODULE(SeparableConv2d);
//
//
// // --- The Main Repeating Block in Xception ---
// struct XceptionBlock : torch::nn::Module {
//     torch::nn::Sequential block, shortcut;
//
//     XceptionBlock(int in_channels, int out_channels, int num_reps, int stride, bool start_with_relu = true) {
//         torch::nn::Sequential layers;
//         if (start_with_relu) {
//             layers->push_back(torch::nn::ReLU());
//         }
//
//         // Add the separable convolutions
//         layers->push_back(SeparableConv2d(in_channels, out_channels, 3, 1, 1));
//         layers->push_back(torch::nn::BatchNorm2d(out_channels));
//         for (int i = 1; i < num_reps; ++i) {
//             layers->push_back(torch::nn::ReLU());
//             layers->push_back(SeparableConv2d(out_channels, out_channels, 3, 1, 1));
//             layers->push_back(torch::nn::BatchNorm2d(out_channels));
//         }
//
//         // The final conv in the block may have a stride
//         if (stride != 1) {
//             layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(stride).padding(1)));
//         }
//         block = register_module("block", layers);
//
//         // Shortcut connection to match dimensions if they change
//         if (stride != 1 || in_channels != out_channels) {
//             shortcut = register_module("shortcut", torch::nn::Sequential(
//                 torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
//                 torch::nn::BatchNorm2d(out_channels)
//             ));
//         }
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto out = block->forward(x);
//         auto short_x = shortcut ? shortcut->forward(x) : x;
//         return out + short_x;
//     }
// };
// TORCH_MODULE(XceptionBlock);
//
//
// // --- The Full Xception Model ---
// struct Xception : torch::nn::Module {
//     torch::nn::Conv2d conv1, conv2;
//     torch::nn::BatchNorm2d bn1, bn2;
//     torch::nn::Sequential entry_flow, middle_flow, exit_flow;
//     torch::nn::Linear fc;
//
//     Xception(int num_middle_blocks, int num_classes = 10) {
//         // --- Entry Flow ---
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(2).padding(1).bias(false)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
//         conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));
//
//         entry_flow = torch::nn::Sequential(
//             XceptionBlock(64, 128, 2, 2, false),
//             XceptionBlock(128, 256, 2, 2),
//             XceptionBlock(256, 728, 2, 2)
//         );
//         register_module("entry_flow", entry_flow);
//
//         // --- Middle Flow ---
//         middle_flow = torch::nn::Sequential();
//         for (int i=0; i < num_middle_blocks; ++i) {
//             middle_flow->push_back(XceptionBlock(728, 728, 3, 1));
//         }
//         register_module("middle_flow", middle_flow);
//
//         // --- Exit Flow ---
//         exit_flow = torch::nn::Sequential(
//             XceptionBlock(728, 1024, 2, 2),
//             SeparableConv2d(1024, 1536, 3, 1, 1),
//             torch::nn::BatchNorm2d(1536),
//             torch::nn::ReLU(),
//             SeparableConv2d(1536, 2048, 3, 1, 1),
//             torch::nn::BatchNorm2d(2048),
//             torch::nn::ReLU()
//         );
//         register_module("exit_flow", exit_flow);
//
//         // --- Classifier ---
//         fc = register_module("fc", torch::nn::Linear(2048, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = torch::relu(bn1(conv1(x)));
//         x = torch::relu(bn2(conv2(x)));
//         x = entry_flow->forward(x);
//         x = middle_flow->forward(x);
//         x = exit_flow->forward(x);
//
//         // Global Average Pooling
//         x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));
//         x = x.view({x.size(0), -1});
//         x = fc->forward(x);
//         return x;
//     }
// };
// TORCH_MODULE(Xception);
//
// // --- GENERIC TRAINING & TESTING FUNCTIONS ---
// template <typename DataLoader>
// void train(Xception& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer,
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
// void test(Xception& model, DataLoader& data_loader, size_t dataset_size, torch::Device device) {
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
//     // --- Hyperparameters ---
//     const int kNumMiddleBlocks = 4; // Use 4 instead of 8 for a smaller model
//     const int64_t kImageSize = 71;   // Upsample MNIST to 71x71
//     const int64_t kTrainBatchSize = 128;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 15;
//     const double kLearningRate = 0.01;
//
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training Xception on " << device << "..." << std::endl;
//     std::cout << "WARNING: Xception is a large model. Training on CPU will be very slow." << std::endl;
//
//     // Model and Optimizer
//     Xception model(kNumMiddleBlocks);
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
//     // Training Loop
//     for (size_t epoch = 1; epoch <= kNumberOfEpochs; ++epoch) {
//         train(model, *train_loader, optimizer, epoch, train_dataset_size, device);
//         test(model, *test_loader, test_dataset_size, device);
//     }
//
//     std::cout << "Training finished." << std::endl;
//     return 0;
// }



namespace xt::models {

    SeparableConv2d::SeparableConv2d(int in_channels, int out_channels, int kernel_size, int stride ,
                                             int padding )
    // Depthwise convolution: processes each channel independently
            : depthwise(torch::nn::Conv2dOptions(in_channels, in_channels, kernel_size)
                                .stride(stride).padding(padding).groups(in_channels).bias(false)),
            // Pointwise convolution: 1x1 conv to mix channel information
              pointwise(torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)) {
        register_module("depthwise", depthwise);
        register_module("pointwise", pointwise);
    }
    auto SeparableConv2d::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }

    torch::Tensor SeparableConv2d::forward(torch::Tensor x) {
        return pointwise(depthwise(x));
    }

    XceptionBlock::XceptionBlock(int in_channels, int out_channels, int num_reps, int stride,
                                         bool start_with_relu ) {
        torch::nn::Sequential layers;
        if (start_with_relu) {
            layers->push_back(torch::nn::ReLU());
        }

        // Add the separable convolutions
        layers->push_back(SeparableConv2d(in_channels, out_channels, 3, 1, 1));
        layers->push_back(torch::nn::BatchNorm2d(out_channels));
        for (int i = 1; i < num_reps; ++i) {
            layers->push_back(torch::nn::ReLU());
            layers->push_back(SeparableConv2d(out_channels, out_channels, 3, 1, 1));
            layers->push_back(torch::nn::BatchNorm2d(out_channels));
        }

        // The final conv in the block may have a stride
        if (stride != 1) {
            layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(stride).padding(1)));
        }
        block = register_module("block", layers);

        // Shortcut connection to match dimensions if they change
        if (stride != 1 || in_channels != out_channels) {
            shortcut = register_module("shortcut", torch::nn::Sequential(
                    torch::nn::Conv2d(
                            torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
                    torch::nn::BatchNorm2d(out_channels)
            ));
        }
    }
    auto XceptionBlock::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }

    torch::Tensor XceptionBlock::forward(torch::Tensor x) {
        auto out = block->forward(x);
        auto short_x = shortcut ? shortcut->forward(x) : x;
        return out + short_x;
    }


    // --- The Full Xception Model ---

    Xception::Xception(int num_middle_blocks, int num_classes ) {
        // --- Entry Flow ---
        conv1 = register_module("conv1",
                                torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(2).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));

        entry_flow = torch::nn::Sequential(
                XceptionBlock(64, 128, 2, 2, false),
                XceptionBlock(128, 256, 2, 2),
                XceptionBlock(256, 728, 2, 2)
        );
        register_module("entry_flow", entry_flow);

        // --- Middle Flow ---
        middle_flow = torch::nn::Sequential();
        for (int i = 0; i < num_middle_blocks; ++i) {
            middle_flow->push_back(XceptionBlock(728, 728, 3, 1));
        }
        register_module("middle_flow", middle_flow);

        // --- Exit Flow ---
        exit_flow = torch::nn::Sequential(
                XceptionBlock(728, 1024, 2, 2),
                SeparableConv2d(1024, 1536, 3, 1, 1),
                torch::nn::BatchNorm2d(1536),
                torch::nn::ReLU(),
                SeparableConv2d(1536, 2048, 3, 1, 1),
                torch::nn::BatchNorm2d(2048),
                torch::nn::ReLU()
        );
        register_module("exit_flow", exit_flow);

        // --- Classifier ---
        fc = register_module("fc", torch::nn::Linear(2048, num_classes));
    }
    auto Xception::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }

    torch::Tensor Xception::forward(torch::Tensor x) {
        x = torch::relu(bn1(conv1(x)));
        x = torch::relu(bn2(conv2(x)));
        x = entry_flow->forward(x);
        x = middle_flow->forward(x);
        x = exit_flow->forward(x);

        // Global Average Pooling
        x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        return x;
    }

//    Xception::Xception(int num_classes, int in_channels)
//    {
//    }
//
//    Xception::Xception(int num_classes, int in_channels, std::vector<int64_t> input_shape)
//    {
//    }
//
//    void Xception::reset()
//    {
//    }
//
//    auto Xception::forward(std::initializer_list<std::any> tensors) -> std::any
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
