#include "include/models/computer_vision/image_classification/zefnet.h"


using namespace std;

//
// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
//
// // --- The ZefNet Model, Adapted for MNIST ---
// // This model follows the spirit of ZefNet (multiple conv stages with LRN)
// // but is adapted for the smaller 28x28 MNIST input.
//
// struct ZefNetImpl : torch::nn::Module {
//     // We separate the model into a feature extractor and a classifier
//     torch::nn::Sequential features, classifier;
//
//     ZefNetImpl(int num_classes = 10) {
//
//         // --- Feature Extractor ---
//         // A sequence of Convolution, ReLU, Pooling, and LRN layers
//         features = torch::nn::Sequential(
//             // Stage 1
//             // In: 1x28x28, Out: 64x14x14
//             // Original ZefNet used 7x7 stride 2. We use 5x5 stride 1 + MaxPool to be gentler on the small input.
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).stride(1).padding(2)),
//             torch::nn::ReLU(),
//             torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
//             // Local Response Normalization was key to ZefNet/AlexNet
//             torch::nn::LocalResponseNorm(torch::nn::LocalResponseNormOptions(5).alpha(0.0001).beta(0.75).k(2.0)),
//
//             // Stage 2
//             // In: 64x14x14, Out: 256x7x7
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 256, 5).stride(1).padding(2)),
//             torch::nn::ReLU(),
//             torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
//             torch::nn::LocalResponseNorm(torch::nn::LocalResponseNormOptions(5).alpha(0.0001).beta(0.75).k(2.0)),
//
//             // Stage 3
//             // In: 256x7x7, Out: 384x7x7
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 384, 3).stride(1).padding(1)),
//             torch::nn::ReLU(),
//
//             // Stage 4
//             // In: 384x7x7, Out: 384x7x7
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 384, 3).stride(1).padding(1)),
//             torch::nn::ReLU(),
//
//             // Stage 5
//             // In: 384x7x7, Out: 256x3x3
//             torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)),
//             torch::nn::ReLU(),
//             torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
//         );
//         register_module("features", features);
//
//         // --- Classifier ---
//         // The original 4096-neuron layers are too large. We use a smaller MLP.
//         // The input size is 256 channels * 3 * 3 spatial dimensions.
//         classifier = torch::nn::Sequential(
//             torch::nn::Dropout(0.5),
//             torch::nn::Linear(256 * 3 * 3, 512),
//             torch::nn::ReLU(),
//             torch::nn::Dropout(0.5),
//             torch::nn::Linear(512, num_classes)
//         );
//         register_module("classifier", classifier);
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = features->forward(x);
//         // Flatten the feature maps before the classifier
//         x = torch::flatten(x, 1);
//         x = classifier->forward(x);
//         return x;
//     }
// };
// TORCH_MODULE(ZefNet);
//
// // --- GENERIC TRAINING & TESTING FUNCTIONS ---
// template <typename DataLoader>
// void train(ZefNet& model, DataLoader& data_loader, torch::optim::Optimizer& optimizer,
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
// void test(ZefNet& model, DataLoader& data_loader, size_t dataset_size, torch::Device device) {
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
//     const int64_t kTrainBatchSize = 128;
//     const int64_t kTestBatchSize = 1000;
//     const int64_t kNumberOfEpochs = 20;
//     const double kLearningRate = 0.001;
//
//     torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
//     std::cout << "Training ZefNet on " << device << "..." << std::endl;
//
//     // Model and Optimizer
//     ZefNet model;
//     model->to(device);
//
//     torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(kLearningRate));
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
//     }
//
//     std::cout << "Training finished." << std::endl;
//     return 0;
// }


namespace xt::models
{
    ZefNet::ZefNet(int num_classes, int in_channels)
    {
    }

    ZefNet::ZefNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void ZefNet::reset()
    {
    }

    auto ZefNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
