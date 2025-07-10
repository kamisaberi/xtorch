#pragma once

#include "../../common.h"


using namespace std;

namespace xt::models {
    // --- The Core Wide-ResNet Block ---
    // This block is different from a standard ResNet block. It uses a (BN-ReLU-Conv) pre-activation
    // sequence and includes a Dropout layer.

    struct WideBasicBlock : xt::Module {
        torch::nn::BatchNorm2d bn1, bn2;
        torch::nn::Conv2d conv1, conv2;
        torch::nn::Dropout dropout;

        // Shortcut connection for residual path
        torch::nn::Sequential shortcut;

        WideBasicBlock(int in_planes, int planes, double dropout_rate, int stride = 1);
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(WideBasicBlock);


    // --- The Full WideResNet Model ---

    struct WideResNet : xt::Module {
        torch::nn::Conv2d conv1;
        torch::nn::Sequential layer1, layer2, layer3;
        torch::nn::BatchNorm2d bn_final;
        torch::nn::Linear linear;

        WideResNet(int depth, int widen_factor, double dropout_rate, int num_classes = 10);

        torch::nn::Sequential _make_layer(int in_planes, int planes, int num_blocks, int stride, double dropout_rate);
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(WideResNet);


//    struct WideResNet : xt::Cloneable<WideResNet>
//    {
//    private:
//
//    public:
//        WideResNet(int num_classes /* classes */, int in_channels = 3/* input channels */);
//
//        WideResNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);
//
//        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//        void reset() override;
//    };
}
