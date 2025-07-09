#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models
{
    // Inception Module
    struct InceptionModule : xt::Module
    {
        InceptionModule(int in_channels, int ch1x1, int ch3x3_reduce, int ch3x3, int ch5x5_reduce, int ch5x5,
                        int pool_proj);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1x1{nullptr}, conv3x3_reduce{nullptr}, conv3x3{nullptr};
        torch::nn::Conv2d conv5x5_reduce{nullptr}, conv5x5{nullptr}, pool_proj{nullptr};
        torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_reduce{nullptr}, bn3x3{nullptr};
        torch::nn::BatchNorm2d bn5x5_reduce{nullptr}, bn5x5{nullptr}, bn_pool{nullptr};
        torch::nn::MaxPool2d pool{nullptr};
    };

    // TORCH_MODULE(InceptionModule);


    // Inception-A Module (Basic Inception block with factorized convolutions)
    struct InceptionAModule : xt::Module
    {
        InceptionAModule(int in_channels, int pool_features);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d branch1x1{nullptr}, branch3x3_1{nullptr}, branch3x3_2{nullptr};
        torch::nn::Conv2d branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
        torch::nn::Conv2d branch_pool_conv{nullptr};
        torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_1{nullptr}, bn3x3_2{nullptr};
        torch::nn::BatchNorm2d bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
        torch::nn::BatchNorm2d bn_pool{nullptr};
        torch::nn::AvgPool2d branch_pool{nullptr};
    };

    // TORCH_MODULE(InceptionAModule);

    // Inception-B Module (Grid size reduction with factorized convolutions)
    struct InceptionBModule : xt::Module
    {
        InceptionBModule(int in_channels);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::MaxPool2d branch_pool{nullptr};
        torch::nn::Conv2d branch3x3_1{nullptr}, branch3x3_2{nullptr}, branch3x3_3{nullptr};
        torch::nn::BatchNorm2d bn3x3_1{nullptr}, bn3x3_2{nullptr}, bn3x3_3{nullptr};
    };

    // TORCH_MODULE(InceptionBModule);

    // Inception-C Module (Asymmetric convolutions: nx1 and 1xn)
    struct InceptionCModule : xt::Module
    {
        InceptionCModule(int in_channels, int channels_7x7);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d branch1x1{nullptr}, branch7x7_1{nullptr}, branch7x7_2{nullptr}, branch7x7_3{nullptr};
        torch::nn::Conv2d branch7x7dbl_1{nullptr}, branch7x7dbl_2{nullptr}, branch7x7dbl_3{nullptr};
        torch::nn::Conv2d branch7x7dbl_4{nullptr}, branch7x7dbl_5{nullptr}, branch_pool_conv{nullptr};
        torch::nn::BatchNorm2d bn1x1{nullptr}, bn7x7_1{nullptr}, bn7x7_2{nullptr}, bn7x7_3{nullptr};
        torch::nn::BatchNorm2d bn7x7dbl_1{nullptr}, bn7x7dbl_2{nullptr}, bn7x7dbl_3{nullptr};
        torch::nn::BatchNorm2d bn7x7dbl_4{nullptr}, bn7x7dbl_5{nullptr}, bn_pool{nullptr};
        torch::nn::AvgPool2d branch_pool{nullptr};
    };

    // TORCH_MODULE(InceptionCModule);


    // Auxiliary Classifier
    struct AuxClassifier : xt::Module
    {
        AuxClassifier(int in_channels, int num_classes);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
        torch::nn::AvgPool2d pool{nullptr};
    };

    // TORCH_MODULE(AuxClassifier);


    // Reduction-A Module
    struct ReductionAModule : xt::Module
    {
        ReductionAModule(int in_channels, int k, int l, int m, int n);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::MaxPool2d branch_pool{nullptr};
        torch::nn::Conv2d branch3x3{nullptr}, branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
        torch::nn::BatchNorm2d bn3x3{nullptr}, bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
    };

    // TORCH_MODULE(ReductionAModule);

    // Reduction-B Module
    struct ReductionBModule : xt::Module
    {
        ReductionBModule(int in_channels);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::MaxPool2d branch_pool{nullptr};
        torch::nn::Conv2d branch3x3_1{nullptr}, branch3x3_2{nullptr};
        torch::nn::Conv2d branch7x7_1{nullptr}, branch7x7_2{nullptr}, branch7x7_3{nullptr}, branch7x7_4{nullptr};
        torch::nn::BatchNorm2d bn3x3_1{nullptr}, bn3x3_2{nullptr};
        torch::nn::BatchNorm2d bn7x7_1{nullptr}, bn7x7_2{nullptr}, bn7x7_3{nullptr}, bn7x7_4{nullptr};
    };

    // TORCH_MODULE(ReductionBModule);


    // InceptionV1 (GoogLeNet)
    struct InceptionV1 : xt::Module
    {
        InceptionV1(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
        torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr}, pool3{nullptr}, pool4{nullptr};
        torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc{nullptr};
        std::shared_ptr<InceptionModule> inception3a{nullptr}, inception3b{nullptr}, inception4a{nullptr};
        std::shared_ptr<InceptionModule> inception4b{nullptr}, inception4c{nullptr}, inception4d{nullptr};
        std::shared_ptr<InceptionModule> inception4e{nullptr}, inception5a{nullptr}, inception5b{nullptr};
    };

    // TORCH_MODULE(InceptionV1);

    // InceptionV2
    struct InceptionV2 : xt::Module
    {
        InceptionV2(int num_classes = 10);
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
        torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
        torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc{nullptr};
        std::shared_ptr<InceptionAModule> inception3a{nullptr}, inception3b{nullptr}, inception3c{nullptr};
        std::shared_ptr<InceptionBModule> inception4a{nullptr}, inception6a{nullptr};
        std::shared_ptr<InceptionAModule> inception5a{nullptr}, inception5b{nullptr}, inception5c{nullptr}, inception5d{
                                              nullptr
                                          };
        std::shared_ptr<InceptionCModule> inception7a{nullptr}, inception7b{nullptr}, inception7c{nullptr};
    };

    // TORCH_MODULE(InceptionV2);


    // InceptionV3
    struct InceptionV3 : xt::Module
    {
        InceptionV3(int num_classes = 10, bool aux_logits = true);

        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x);

        bool aux_logits_;
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
        torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
        torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc{nullptr};
        std::shared_ptr<InceptionAModule> inception_a1{nullptr}, inception_a2{nullptr}, inception_a3{nullptr};
        std::shared_ptr<InceptionBModule> inception_b{nullptr}, inception_d{nullptr};
        std::shared_ptr<InceptionAModule> inception_c1{nullptr}, inception_c2{nullptr}, inception_c3{nullptr},
                                          inception_c4{nullptr};
        std::shared_ptr<InceptionCModule> inception_e1{nullptr}, inception_e2{nullptr};
        std::shared_ptr<AuxClassifier> aux_classifier{nullptr};
    };

    // TORCH_MODULE(InceptionV3);

    // InceptionV4
    struct InceptionV4 : xt::Module
    {
        InceptionV4(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
        torch::nn::Conv2d conv5{nullptr}, conv6{nullptr}, conv7{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};
        torch::nn::BatchNorm2d bn5{nullptr}, bn6{nullptr}, bn7{nullptr};
        torch::nn::MaxPool2d pool1{nullptr};
        torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc{nullptr};
        std::shared_ptr<InceptionAModule> inception_a1{nullptr}, inception_a2{nullptr}, inception_a3{nullptr},
                                          inception_a4{nullptr};
        std::shared_ptr<ReductionAModule> reduction_a{nullptr};
        std::shared_ptr<InceptionBModule> inception_b1{nullptr}, inception_b2{nullptr}, inception_b3{nullptr};
        std::shared_ptr<InceptionBModule> inception_b4{nullptr}, inception_b5{nullptr}, inception_b6{nullptr},
                                          inception_b7{nullptr};
        std::shared_ptr<ReductionBModule> reduction_b{nullptr};
        std::shared_ptr<InceptionCModule> inception_c1{nullptr}, inception_c2{nullptr}, inception_c3{nullptr};
    };

    // TORCH_MODULE(InceptionV4);


    // struct InceptionV1 : xt::Cloneable<InceptionV1>
    // {
    // private:
    //
    // public:
    //     InceptionV1(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     InceptionV1(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
    //
    // struct InceptionV2 : xt::Cloneable<InceptionV2>
    // {
    // private:
    //
    // public:
    //     InceptionV2(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     InceptionV2(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
    //
    // struct InceptionV3 : xt::Cloneable<InceptionV3>
    // {
    // private:
    //
    // public:
    //     InceptionV3(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     InceptionV3(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
    //
    // struct InceptionV4 : xt::Cloneable<InceptionV4>
    // {
    // private:
    //
    // public:
    //     InceptionV4(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     InceptionV4(int num_classes, int in_channels, std::vector<int64_t> input_shape);
    //
    //     auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
}
