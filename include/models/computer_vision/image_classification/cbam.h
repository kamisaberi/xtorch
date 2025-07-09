#pragma once

#include "../../common.h"


using namespace std;

namespace xt::models
{
    // Channel Attention Module
    struct ChannelAttention : xt::Module
    {
        ChannelAttention(int channels, int reduction = 16);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };

    // TORCH_MODULE(ChannelAttention);

    // Spatial Attention Module
    struct SpatialAttention : xt::Module
    {
        SpatialAttention(int channels);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv{nullptr};
    };

    // TORCH_MODULE(SpatialAttention);

    // CBAM Module
    struct CBAM : xt::Module
    {
        CBAM(int channels, int reduction = 16);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        ChannelAttention channel_attention{nullptr};
        SpatialAttention spatial_attention{nullptr};
    };

    // TORCH_MODULE(CBAM);

    // CNN with CBAM
    struct CBAMNet : torch::nn::Module
    {
        CBAMNet(int in_channels, int num_classes);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
        CBAM cbam1{nullptr}, cbam2{nullptr}, cbam3{nullptr};
        torch::nn::AdaptiveAvgPool2d pool{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(CBAMNet);


    // struct CBAM : xt::Cloneable<CBAM>
    // {
    // protected:
    //
    // public:
    //     explicit CBAM(int num_classes/* classes */, int in_channels = 1/*  input channels */);
    //
    //     CBAM(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
}
