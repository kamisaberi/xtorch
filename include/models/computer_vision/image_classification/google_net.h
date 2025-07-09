#pragma once

#include "../../common.h"


namespace xt::models
{
    // Inception Module
    struct InceptionModule : xt::Module
    {
        InceptionModule(int in_channels, int ch1x1, int ch3x3red, int ch3x3, int ch5x5red, int ch5x5,
                        int pool_proj_count);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1x1{nullptr}, conv3x3_reduce{nullptr}, conv3x3{nullptr};
        torch::nn::Conv2d conv5x5_reduce{nullptr}, conv5x5{nullptr}, pool_proj{nullptr};
        torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_reduce{nullptr}, bn3x3{nullptr};
        torch::nn::BatchNorm2d bn5x5_reduce{nullptr}, bn5x5{nullptr}, bn_pool_proj{nullptr};
        torch::nn::MaxPool2d pool{nullptr};
    };

    // TORCH_MODULE(InceptionModule);

    // Sified GoogLeNet
    struct GoogLeNet : xt::Module
    {
        GoogLeNet(int in_channels, int num_classes);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr};
        torch::nn::BatchNorm2d stem_bn{nullptr};
        std::shared_ptr<InceptionModule> inception1{nullptr}, inception2{nullptr};
        torch::nn::MaxPool2d pool{nullptr};
        torch::nn::AdaptiveAvgPool2d global_pool{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(GoogLeNet);


    // struct GoogLeNet : xt::Cloneable<GoogLeNet> {
    // private:
    //
    // public:
    //     GoogLeNet(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     GoogLeNet(int num_classes, int in_channels, std::vector <int64_t> input_shape);
    //
    //     auto forward(std::initializer_list <std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
}
