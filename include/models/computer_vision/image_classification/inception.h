#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models {

    // Inception Module
    struct InceptionModuleImpl : torch::nn::Module {
        InceptionModuleImpl(int in_channels, int ch1x1, int ch3x3_reduce, int ch3x3, int ch5x5_reduce, int ch5x5,
                            int pool_proj);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1x1{nullptr}, conv3x3_reduce{nullptr}, conv3x3{nullptr};
        torch::nn::Conv2d conv5x5_reduce{nullptr}, conv5x5{nullptr}, pool_proj{nullptr};
        torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_reduce{nullptr}, bn3x3{nullptr};
        torch::nn::BatchNorm2d bn5x5_reduce{nullptr}, bn5x5{nullptr}, bn_pool{nullptr};
        torch::nn::MaxPool2d pool{nullptr};
    };

    TORCH_MODULE(InceptionModule);

    // InceptionV1 (GoogLeNet)
    struct InceptionV1Impl : torch::nn::Module {
        InceptionV1Impl(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
        torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr}, pool3{nullptr}, pool4{nullptr};
        torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc{nullptr};
        InceptionModule inception3a{nullptr}, inception3b{nullptr}, inception4a{nullptr};
        InceptionModule inception4b{nullptr}, inception4c{nullptr}, inception4d{nullptr};
        InceptionModule inception4e{nullptr}, inception5a{nullptr}, inception5b{nullptr};
    };

    TORCH_MODULE(InceptionV1);


    struct InceptionV1 : xt::Cloneable<InceptionV1> {
    private:

    public:
        InceptionV1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV1(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };

    struct InceptionV2 : xt::Cloneable<InceptionV2> {
    private:

    public:
        InceptionV2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV2(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };

    struct InceptionV3 : xt::Cloneable<InceptionV3> {
    private:

    public:
        InceptionV3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV3(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };

    struct InceptionV4 : xt::Cloneable<InceptionV4> {
    private:

    public:
        InceptionV4(int num_classes /* classes */, int in_channels = 3/* input channels */);

        InceptionV4(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


}
