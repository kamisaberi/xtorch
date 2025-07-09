#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models
{
    // Dense Layer (Bottleneck: 1x1 conv -> 3x3 conv)
    struct DenseLayer : xt::Module
    {
        DenseLayer(int in_channels, int growth_rate);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    };

    // TORCH_MODULE(DenseLayer);

    // Dense Block
    struct DenseBlock : xt::Module
    {
        DenseBlock(int num_layers, int in_channels, int growth_rate);
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        vector<std::shared_ptr<xt::Module>> layers;
    };

    // TORCH_MODULE(DenseBlock);

    // Transition Layer (1x1 conv + 2x2 avg pool)
    struct TransitionLayer : xt::Module
    {
        TransitionLayer(int in_channels, int out_channels);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        torch::Tensor forward(torch::Tensor x);

        torch::nn::BatchNorm2d bn{nullptr};
        torch::nn::Conv2d conv{nullptr};
        torch::nn::AvgPool2d pool{nullptr};
    };

    // TORCH_MODULE(TransitionLayer);

    // DenseNet121
    struct DenseNet121 : xt::Module
    {
        DenseNet121(int num_classes = 10, int growth_rate = 32, int init_channels = 64);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv0{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
        std::shared_ptr<DenseBlock> dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
        std::shared_ptr<TransitionLayer> trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(DenseNet121);


    // DenseNet169
    struct DenseNet169 : xt::Module
    {
        DenseNet169(int num_classes = 10, int growth_rate = 32, int init_channels = 64);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv0{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
        std::shared_ptr<DenseBlock> dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
        std::shared_ptr<TransitionLayer> trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(DenseNet169);


    // DenseNet201
    struct DenseNet201 : xt::Module
    {
        DenseNet201(int num_classes = 10, int growth_rate = 32, int init_channels = 64);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv0{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
        std::shared_ptr<DenseBlock> dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
        std::shared_ptr<TransitionLayer> trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(DenseNet201);


    // DenseNet264
    struct DenseNet264 : xt::Module
    {
        DenseNet264(int num_classes = 10, int growth_rate = 32, int init_channels = 64);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv0{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
        std::shared_ptr<DenseBlock> dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
        std::shared_ptr<TransitionLayer> trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    // TORCH_MODULE(DenseNet264);


    // struct DenseNet121 : xt::Cloneable<DenseNet121> {
    // private:
    //
    // public:
    //     DenseNet121(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     DenseNet121(int num_classes, int in_channels, std::vector <int64_t> input_shape);
    //
    //     auto forward(std::initializer_list <std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
    //
    //
    // struct DenseNet169 : xt::Cloneable<DenseNet169> {
    // private:
    //
    // public:
    //     DenseNet169(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     DenseNet169(int num_classes, int in_channels, std::vector <int64_t> input_shape);
    //
    //     auto forward(std::initializer_list <std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
    //
    //
    // struct DenseNet201 : xt::Cloneable<DenseNet201> {
    // private:
    //
    // public:
    //     DenseNet201(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     DenseNet201(int num_classes, int in_channels, std::vector <int64_t> input_shape);
    //
    //     auto forward(std::initializer_list <std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
    //
    //
    // struct DenseNet264 : xt::Cloneable<DenseNet264> {
    // private:
    //
    // public:
    //     DenseNet264(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     DenseNet264(int num_classes, int in_channels, std::vector <int64_t> input_shape);
    //
    //     auto forward(std::initializer_list <std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
}
