#pragma once
#include "../../common.h"

namespace xt::models
{
    // --- The Core Squeeze-and-Excitation Module ---

    struct SEModule1 : xt::Module
    {
        torch::nn::AdaptiveAvgPool2d squeeze;
        torch::nn::Sequential excitation;

        SEModule1(int in_channels, int reduction_ratio = 16);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(SEModule);


    // --- A Basic ResNet Block with an integrated SE Module ---

    struct SEBasicBlock : xt::Module
    {
        torch::nn::Conv2d conv1, conv2;
        torch::nn::BatchNorm2d bn1, bn2;
        std::shared_ptr<SEModule1> se_module;

        // Shortcut for residual connection
        torch::nn::Sequential shortcut;

        SEBasicBlock(int in_planes, int planes, int stride = 1, int reduction_ratio = 16);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(SEBasicBlock);


    // --- The Full SENet Model (using the SE-ResNet backbone) ---

    struct SENet : xt::Module
    {
        torch::nn::Conv2d conv1;
        torch::nn::BatchNorm2d bn1;
        torch::nn::Sequential layer1, layer2, layer3;
        torch::nn::Linear linear;

        SENet(const std::vector<int>& num_blocks, int num_classes = 10, int reduction_ratio = 16);
        torch::nn::Sequential _make_layer(int in_planes, int planes, int num_blocks, int stride, int reduction);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(SENet);

    //    struct SENet : xt::Cloneable<SENet>
    //    {
    //    private:
    //
    //    public:
    //        SENet(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //        SENet(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //        void reset() override;
    //    };
}
