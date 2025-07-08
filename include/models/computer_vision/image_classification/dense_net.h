#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models {

    // Dense Layer (Bottleneck: 1x1 conv -> 3x3 conv)
    struct DenseLayerImpl : torch::nn::Module {
        DenseLayerImpl(int in_channels, int growth_rate);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    };

    TORCH_MODULE(DenseLayer);

    // Dense Block
    struct DenseBlockImpl : torch::nn::Module {
        DenseBlockImpl(int num_layers, int in_channels, int growth_rate);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::ModuleList layers{torch::nn::ModuleList()};
    };

    TORCH_MODULE(DenseBlock);

    // Transition Layer (1x1 conv + 2x2 avg pool)
    struct TransitionLayerImpl : torch::nn::Module {
        TransitionLayerImpl(int in_channels, int out_channels);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::BatchNorm2d bn{nullptr};
        torch::nn::Conv2d conv{nullptr};
        torch::nn::AvgPool2d pool{nullptr};
    };

    TORCH_MODULE(TransitionLayer);

    // DenseNet121
    struct DenseNet121Impl : torch::nn::Module {
        DenseNet121Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv0{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
        DenseBlock dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
        TransitionLayer trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    TORCH_MODULE(DenseNet121);


    // DenseNet169
    struct DenseNet169Impl : torch::nn::Module {
        DenseNet169Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv0{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
        DenseBlock dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
        TransitionLayer trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    TORCH_MODULE(DenseNet169);


    struct DenseNet121 : xt::Cloneable<DenseNet121> {
    private:

    public:
        DenseNet121(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet121(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    struct DenseNet169 : xt::Cloneable<DenseNet169> {
    private:

    public:
        DenseNet169(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet169(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    struct DenseNet201 : xt::Cloneable<DenseNet201> {
    private:

    public:
        DenseNet201(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet201(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    struct DenseNet264 : xt::Cloneable<DenseNet264> {
    private:

    public:
        DenseNet264(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet264(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };
}
