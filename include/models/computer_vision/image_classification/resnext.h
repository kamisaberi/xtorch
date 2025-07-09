#pragma once

#include "../../common.h"


using namespace std;

namespace xt::models {
    // --- The Core ResNeXt Block ---

    struct ResNeXtBlockImpl : torch::nn::Module {
        // A ResNeXt block splits the input into 'cardinality' groups,
        // processes each with a small bottleneck, and then merges them.
        // This is implemented efficiently using a grouped convolution.
        torch::nn::Conv2d conv1, conv2_grouped, conv3;
        torch::nn::BatchNorm2d bn1, bn2, bn3;

        // Shortcut connection to handle dimension changes
        torch::nn::Sequential shortcut;

        ResNeXtBlockImpl(int in_channels, int out_channels, int stride, int cardinality, int bottleneck_width);

        torch::Tensor forward(torch::Tensor x);
    };

    TORCH_MODULE(ResNeXtBlock);


    // --- The Full ResNeXt Model ---

    struct ResNeXtImpl : torch::nn::Module {
        torch::nn::Conv2d conv1;
        torch::nn::BatchNorm2d bn1;
        torch::nn::Sequential layer1, layer2, layer3;
        torch::nn::Linear linear;

        ResNeXtImpl(int num_blocks1, int num_blocks2, int num_blocks3, int cardinality, int bottleneck_width,
                    int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);
    };

    TORCH_MODULE(ResNeXt);


//    struct ResNeXt : xt::Cloneable<ResNeXt>
//    {
//    private:
//
//    public:
//        ResNeXt(int num_classes /* classes */, int in_channels = 3/* input channels */);
//
//        ResNeXt(int num_classes, int in_channels, std::vector<int64_t> input_shape);
//
//        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//        void reset() override;
//    };
}
