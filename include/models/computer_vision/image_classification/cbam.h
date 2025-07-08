#pragma once

#include "../../common.h"


using namespace std;

namespace xt::models {

    // Channel Attention Module
    struct ChannelAttentionImpl : torch::nn::Module {
        ChannelAttentionImpl(int channels, int reduction = 16) {
            fc1 = register_module("fc1", torch::nn::Linear(channels, channels / reduction));
            fc2 = register_module("fc2", torch::nn::Linear(channels / reduction, channels));
        }

        torch::Tensor forward(torch::Tensor x) {
            // x: [batch, channels, h, w]
            auto avg_pool = torch::avg_pool2d(x, {x.size(2), x.size(3)}).squeeze(-1).squeeze(-1); // [batch, channels]
            auto max_pool = torch::max_pool2d(x, {x.size(2), x.size(3)}).squeeze(-1).squeeze(-1); // [batch, channels]

            // Shared MLP
            auto avg_out = torch::relu(fc1->forward(avg_pool));
            avg_out = torch::sigmoid(fc2->forward(avg_out)); // [batch, channels]
            auto max_out = torch::relu(fc1->forward(max_pool));
            max_out = torch::sigmoid(fc2->forward(max_out)); // [batch, channels]

            auto attention = (avg_out + max_out).unsqueeze(-1).unsqueeze(-1); // [batch, channels, 1, 1]
            return x * attention; // Element-wise multiplication
        }

        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };

    TORCH_MODULE(ChannelAttention);

    // Spatial Attention Module
    struct SpatialAttentionImpl : torch::nn::Module {
        SpatialAttentionImpl(int channels) {
            conv = register_module("conv", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(2, 1, 7).stride(1).padding(3)));
        }

        torch::Tensor forward(torch::Tensor x) {
            // x: [batch, channels, h, w]
            auto avg_pool = x.mean(1, true); // [batch, 1, h, w]
            auto max_pool = x.max(1, true).values; // [batch, 1, h, w]
            auto concat = torch::cat({avg_pool, max_pool}, 1); // [batch, 2, h, w]
            auto attention = torch::sigmoid(conv->forward(concat)); // [batch, 1, h, w]
            return x * attention; // Element-wise multiplication
        }

        torch::nn::Conv2d conv{nullptr};
    };

    TORCH_MODULE(SpatialAttention);

    // CBAM Module
    struct CBAMImpl : torch::nn::Module {
        CBAMImpl(int channels, int reduction = 16) {
            channel_attention = register_module("channel_attention", ChannelAttention(channels, reduction));
            spatial_attention = register_module("spatial_attention", SpatialAttention(channels));
        }

        torch::Tensor forward(torch::Tensor x) {
            // Sequential attention: channel then spatial
            x = channel_attention->forward(x);
            x = spatial_attention->forward(x);
            return x;
        }

        ChannelAttention channel_attention{nullptr};
        SpatialAttention spatial_attention{nullptr};
    };

    TORCH_MODULE(CBAM);

    // CNN with CBAM
    struct CBAMNetImpl : torch::nn::Module {
        CBAMNetImpl(int in_channels, int num_classes) {
            conv1 = register_module("conv1", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
            cbam1 = register_module("cbam1", CBAM(64));
            conv2 = register_module("conv2", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
            cbam2 = register_module("cbam2", CBAM(128));
            conv3 = register_module("conv3", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));
            bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
            cbam3 = register_module("cbam3", CBAM(256));
            pool = register_module("pool", torch::nn::AdaptiveAvgPool2d(1));
            fc = register_module("fc", torch::nn::Linear(256, num_classes));
        }

        torch::Tensor forward(torch::Tensor x) {
            // x: [batch, in_channels, 32, 32]
            x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 64, 32, 32]
            x = cbam1->forward(x);
            x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 128, 16, 16]
            x = cbam2->forward(x);
            x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 256, 8, 8]
            x = cbam3->forward(x);
            x = pool->forward(x).view({x.size(0), -1}); // [batch, 256]
            x = fc->forward(x); // [batch, num_classes]
            return x;
        }

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
        CBAM cbam1{nullptr}, cbam2{nullptr}, cbam3{nullptr};
        torch::nn::AdaptiveAvgPool2d pool{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    TORCH_MODULE(CBAMNet);


    struct CBAM : xt::Cloneable<CBAM> {
    protected:

    public:
        explicit CBAM(int num_classes/* classes */, int in_channels = 1/*  input channels */);

        CBAM(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };
}
