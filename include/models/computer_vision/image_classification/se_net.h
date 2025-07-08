#pragma once
#include "../../common.h"

namespace xt::models
{
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

    struct SENet : xt::Cloneable<SENet>
    {
    private:

    public:
        SENet(int num_classes /* classes */, int in_channels = 3/* input channels */);

        SENet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
