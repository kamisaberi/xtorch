#pragma once
#include "../../common.h"

namespace xt::models
{
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


    struct NetworkInNetwork : xt::Cloneable<NetworkInNetwork>
    {
    private:

    public:
        NetworkInNetwork(int num_classes /* classes */, int in_channels = 3/* input channels */);

        NetworkInNetwork(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        void reset() override;
    };
}
