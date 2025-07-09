#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models
{
    struct BasicConv2d : xt::Module
    {
        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
        bool use_relu;

        BasicConv2d(int in_planes, int out_planes, int kernel_size, int stride, int padding, bool relu = true);

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(BasicConv2d);


    // Inception-ResNet-A block
    struct InceptionResNetA : xt::Module
    {
        std::shared_ptr<BasicConv2d> b0, b1_0, b1_1, b2_0, b2_1, b2_2;
        torch::nn::Conv2d conv;
        double scale;

        InceptionResNetA(int in_planes, double scale = 1.0);
        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(InceptionResNetA);

    // Inception-ResNet-B block
    struct InceptionResNetB : xt::Module
    {
        std::shared_ptr<BasicConv2d> b0, b1_0, b1_1, b1_2;
        torch::nn::Conv2d conv;
        double scale;

        InceptionResNetB(int in_planes, double scale = 1.0);
        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(InceptionResNetB);

    // Inception-ResNet-C block
    struct InceptionResNetC : xt::Module
    {
        std::shared_ptr<BasicConv2d> b0, b1_0, b1_1, b1_2;
        torch::nn::Conv2d conv;
        double scale;

        InceptionResNetC(int in_planes, double scale = 1.0);
        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(InceptionResNetC);

    // Reduction-A block
    struct ReductionA : xt::Module
    {
        std::shared_ptr<BasicConv2d> b0, b1_0, b1_1, b1_2;
        torch::nn::MaxPool2d b2;

        ReductionA(int in_planes, int k, int l, int m, int n);
        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(ReductionA);

    // Reduction-B block
    struct ReductionB : xt::Module
    {
        std::shared_ptr<BasicConv2d> b0_0, b0_1, b1_0, b1_1, b2_0, b2_1, b2_2;
        torch::nn::MaxPool2d b3;

        ReductionB(int in_planes);

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(ReductionB);


    // The complex stem of the InceptionResNetV2.
    struct Stem : xt::Module
    {
        std::shared_ptr<BasicConv2d> conv2d_1a, conv2d_2a, conv2d_2b;
        torch::nn::MaxPool2d maxpool_3a;
        std::shared_ptr<BasicConv2d> conv2d_3b, conv2d_4a;
        std::shared_ptr<BasicConv2d> branch_0_conv, branch_1_conv_1, branch_1_conv_2;
        torch::nn::MaxPool2d branch_pool;

        Stem(int in_channels);

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(Stem);


    // The complete InceptionResNetV1 model, adapted for MNIST
    struct InceptionResNetV1 : xt::Module
    {
        std::shared_ptr<BasicConv2d> conv2d_1a, conv2d_2a, conv2d_2b;
        torch::nn::MaxPool2d maxpool_3a;
        std::shared_ptr<BasicConv2d> conv2d_3b, conv2d_4a;
        torch::nn::MaxPool2d maxpool_5a;
        torch::nn::Sequential repeat, repeat_1, repeat_2;
        std::shared_ptr<ReductionA> mixed_6a;
        std::shared_ptr<ReductionB> mixed_7a;
        std::shared_ptr<BasicConv2d> block8;
        torch::nn::AdaptiveAvgPool2d avgpool_1a;
        torch::nn::Dropout dropout;
        torch::nn::Linear logits;

        InceptionResNetV1(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(InceptionResNetV1);


    // The Full InceptionResNetV2 model adapted for MNIST
    struct InceptionResNetV2 : xt::Module
    {
        std::shared_ptr<Stem> stem;
        torch::nn::Sequential repeat_a, repeat_b, repeat_c;
        std::shared_ptr<ReductionA> reduction_a;
        std::shared_ptr<ReductionB> reduction_b;
        torch::nn::AdaptiveAvgPool2d avgpool;
        torch::nn::Dropout dropout;
        torch::nn::Linear logits;

        InceptionResNetV2(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(InceptionResNetV2);


    // struct InceptionResNetV1 : xt::Cloneable<InceptionResNetV1>
    // {
    // private:
    //
    // public:
    //     InceptionResNetV1(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     InceptionResNetV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
    //
    // struct InceptionResNetV2 : xt::Cloneable<InceptionResNetV2>
    // {
    // private:
    //
    // public:
    //     InceptionResNetV2(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     InceptionResNetV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
}
