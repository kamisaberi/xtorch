#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models
{
    // Inception Module
    struct InceptionModuleImpl : torch::nn::Module
    {
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


    // Inception-A Module (Basic Inception block with factorized convolutions)
    struct InceptionAModuleImpl : torch::nn::Module
    {
        InceptionAModuleImpl(int in_channels, int pool_features);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d branch1x1{nullptr}, branch3x3_1{nullptr}, branch3x3_2{nullptr};
        torch::nn::Conv2d branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
        torch::nn::Conv2d branch_pool_conv{nullptr};
        torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_1{nullptr}, bn3x3_2{nullptr};
        torch::nn::BatchNorm2d bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
        torch::nn::BatchNorm2d bn_pool{nullptr};
        torch::nn::AvgPool2d branch_pool{nullptr};
    };

    TORCH_MODULE(InceptionAModule);

    // Inception-B Module (Grid size reduction with factorized convolutions)
    struct InceptionBModuleImpl : torch::nn::Module
    {
        InceptionBModuleImpl(int in_channels);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::MaxPool2d branch_pool{nullptr};
        torch::nn::Conv2d branch3x3_1{nullptr}, branch3x3_2{nullptr}, branch3x3_3{nullptr};
        torch::nn::BatchNorm2d bn3x3_1{nullptr}, bn3x3_2{nullptr}, bn3x3_3{nullptr};
    };

    TORCH_MODULE(InceptionBModule);

    // Inception-C Module (Asymmetric convolutions: nx1 and 1xn)
    struct InceptionCModuleImpl : torch::nn::Module
    {
        InceptionCModuleImpl(int in_channels, int channels_7x7);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d branch1x1{nullptr}, branch7x7_1{nullptr}, branch7x7_2{nullptr}, branch7x7_3{nullptr};
        torch::nn::Conv2d branch7x7dbl_1{nullptr}, branch7x7dbl_2{nullptr}, branch7x7dbl_3{nullptr};
        torch::nn::Conv2d branch7x7dbl_4{nullptr}, branch7x7dbl_5{nullptr}, branch_pool_conv{nullptr};
        torch::nn::BatchNorm2d bn1x1{nullptr}, bn7x7_1{nullptr}, bn7x7_2{nullptr}, bn7x7_3{nullptr};
        torch::nn::BatchNorm2d bn7x7dbl_1{nullptr}, bn7x7dbl_2{nullptr}, bn7x7dbl_3{nullptr};
        torch::nn::BatchNorm2d bn7x7dbl_4{nullptr}, bn7x7dbl_5{nullptr}, bn_pool{nullptr};
        torch::nn::AvgPool2d branch_pool{nullptr};
    };

    TORCH_MODULE(InceptionCModule);


    // Auxiliary Classifier
    struct AuxClassifierImpl : torch::nn::Module
    {
        AuxClassifierImpl(int in_channels, int num_classes)
        {
            conv = register_module("conv", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(in_channels, 128, 1).bias(false)));
            bn = register_module("bn", torch::nn::BatchNorm2d(128));
            fc1 = register_module("fc1", torch::nn::Linear(128 * 4 * 4, 1024));
            fc2 = register_module("fc2", torch::nn::Linear(1024, num_classes));
            pool = register_module("pool", torch::nn::AvgPool2d(
                                       torch::nn::AvgPool2dOptions(5).stride(3)));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            x = pool->forward(x);
            x = torch::relu(bn->forward(conv->forward(x)));
            x = x.view({x.size(0), -1});
            x = torch::relu(fc1->forward(x));
            x = fc2->forward(x);
            return x;
        }

        torch::nn::Conv2d conv{nullptr};
        torch::nn::BatchNorm2d bn{nullptr};
        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
        torch::nn::AvgPool2d pool{nullptr};
    };

    TORCH_MODULE(AuxClassifier);


    // Reduction-A Module
    struct ReductionAModuleImpl : torch::nn::Module
    {
        ReductionAModuleImpl(int in_channels, int k, int l, int m, int n)
        {
            // Branch 1: 3x3 max pool
            branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
                                              torch::nn::MaxPool2dOptions(3).stride(2)));

            // Branch 2: 3x3 conv (stride 2)
            branch3x3 = register_module("branch3x3", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(in_channels, n, 3).stride(2).bias(false)));
            bn3x3 = register_module("bn3x3", torch::nn::BatchNorm2d(n));

            // Branch 3: 1x1 conv -> 3x3 conv -> 3x3 conv (stride 2)
            branch3x3dbl_1 = register_module("branch3x3dbl_1", torch::nn::Conv2d(
                                                 torch::nn::Conv2dOptions(in_channels, k, 1).bias(false)));
            bn3x3dbl_1 = register_module("bn3x3dbl_1", torch::nn::BatchNorm2d(k));
            branch3x3dbl_2 = register_module("branch3x3dbl_2", torch::nn::Conv2d(
                                                 torch::nn::Conv2dOptions(k, l, 3).padding(1).bias(false)));
            bn3x3dbl_2 = register_module("bn3x3dbl_2", torch::nn::BatchNorm2d(l));
            branch3x3dbl_3 = register_module("branch3x3dbl_3", torch::nn::Conv2d(
                                                 torch::nn::Conv2dOptions(l, m, 3).stride(2).bias(false)));
            bn3x3dbl_3 = register_module("bn3x3dbl_3", torch::nn::BatchNorm2d(m));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            auto branch1 = branch_pool->forward(x);
            auto branch2 = torch::relu(bn3x3->forward(branch3x3->forward(x)));
            auto branch3 = torch::relu(bn3x3dbl_1->forward(branch3x3dbl_1->forward(x)));
            branch3 = torch::relu(bn3x3dbl_2->forward(branch3x3dbl_2->forward(branch3)));
            branch3 = torch::relu(bn3x3dbl_3->forward(branch3x3dbl_3->forward(branch3)));
            return torch::cat({branch1, branch2, branch3}, 1);
        }

        torch::nn::MaxPool2d branch_pool{nullptr};
        torch::nn::Conv2d branch3x3{nullptr}, branch3x3dbl_1{nullptr}, branch3x3dbl_2{nullptr}, branch3x3dbl_3{nullptr};
        torch::nn::BatchNorm2d bn3x3{nullptr}, bn3x3dbl_1{nullptr}, bn3x3dbl_2{nullptr}, bn3x3dbl_3{nullptr};
    };

    TORCH_MODULE(ReductionAModule);

    // Reduction-B Module
    struct ReductionBModuleImpl : torch::nn::Module
    {
        ReductionBModuleImpl(int in_channels)
        {
            // Branch 1: 3x3 max pool
            branch_pool = register_module("branch_pool", torch::nn::MaxPool2d(
                                              torch::nn::MaxPool2dOptions(3).stride(2)));

            // Branch 2: 1x1 conv -> 3x3 conv (stride 2)
            branch3x3_1 = register_module("branch3x3_1", torch::nn::Conv2d(
                                              torch::nn::Conv2dOptions(in_channels, 192, 1).bias(false)));
            bn3x3_1 = register_module("bn3x3_1", torch::nn::BatchNorm2d(192));
            branch3x3_2 = register_module("branch3x3_2", torch::nn::Conv2d(
                                              torch::nn::Conv2dOptions(192, 192, 3).stride(2).bias(false)));
            bn3x3_2 = register_module("bn3x3_2", torch::nn::BatchNorm2d(192));

            // Branch 3: 1x1 conv -> 1x7 conv -> 7x1 conv -> 3x3 conv (stride 2)
            branch7x7_1 = register_module("branch7x7_1", torch::nn::Conv2d(
                                              torch::nn::Conv2dOptions(in_channels, 256, 1).bias(false)));
            bn7x7_1 = register_module("bn7x7_1", torch::nn::BatchNorm2d(256));
            branch7x7_2 = register_module("branch7x7_2", torch::nn::Conv2d(
                                              torch::nn::Conv2dOptions(256, 256, {1, 7}).padding({0, 3}).bias(false)));
            bn7x7_2 = register_module("bn7x7_2", torch::nn::BatchNorm2d(256));
            branch7x7_3 = register_module("branch7x7_3", torch::nn::Conv2d(
                                              torch::nn::Conv2dOptions(256, 320, {7, 1}).padding({3, 0}).bias(false)));
            bn7x7_3 = register_module("bn7x7_3", torch::nn::BatchNorm2d(320));
            branch7x7_4 = register_module("branch7x7_4", torch::nn::Conv2d(
                                              torch::nn::Conv2dOptions(320, 320, 3).stride(2).bias(false)));
            bn7x7_4 = register_module("bn7x7_4", torch::nn::BatchNorm2d(320));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            auto branch1 = branch_pool->forward(x);
            auto branch2 = torch::relu(bn3x3_1->forward(branch3x3_1->forward(x)));
            branch2 = torch::relu(bn3x3_2->forward(branch3x3_2->forward(branch2)));
            auto branch3 = torch::relu(bn7x7_1->forward(branch7x7_1->forward(x)));
            branch3 = torch::relu(bn7x7_2->forward(branch7x7_2->forward(branch3)));
            branch3 = torch::relu(bn7x7_3->forward(branch7x7_3->forward(branch3)));
            branch3 = torch::relu(bn7x7_4->forward(branch7x7_4->forward(branch3)));
            return torch::cat({branch1, branch2, branch3}, 1);
        }

        torch::nn::MaxPool2d branch_pool{nullptr};
        torch::nn::Conv2d branch3x3_1{nullptr}, branch3x3_2{nullptr};
        torch::nn::Conv2d branch7x7_1{nullptr}, branch7x7_2{nullptr}, branch7x7_3{nullptr}, branch7x7_4{nullptr};
        torch::nn::BatchNorm2d bn3x3_1{nullptr}, bn3x3_2{nullptr};
        torch::nn::BatchNorm2d bn7x7_1{nullptr}, bn7x7_2{nullptr}, bn7x7_3{nullptr}, bn7x7_4{nullptr};
    };

    TORCH_MODULE(ReductionBModule);


    // InceptionV1 (GoogLeNet)
    struct InceptionV1Impl : torch::nn::Module
    {
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

    // InceptionV2
    struct InceptionV2Impl : torch::nn::Module
    {
        InceptionV2Impl(int num_classes = 10);
        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
        torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
        torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc{nullptr};
        InceptionAModule inception3a{nullptr}, inception3b{nullptr}, inception3c{nullptr};
        InceptionBModule inception4a{nullptr}, inception6a{nullptr};
        InceptionAModule inception5a{nullptr}, inception5b{nullptr}, inception5c{nullptr}, inception5d{nullptr};
        InceptionCModule inception7a{nullptr}, inception7b{nullptr}, inception7c{nullptr};
    };

    TORCH_MODULE(InceptionV2);


    // InceptionV3
    struct InceptionV3Impl : torch::nn::Module
    {
        InceptionV3Impl(int num_classes = 10, bool aux_logits = true) : aux_logits_(aux_logits)
        {
            // Stem
            conv1 = register_module("conv1", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false)));
            // Simplified stride
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
            conv2 = register_module("conv2", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(32, 32, 3).padding(1).bias(false)));
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(32));
            conv3 = register_module("conv3", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
            bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
            pool1 = register_module("pool1", torch::nn::MaxPool2d(
                                        torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
            conv4 = register_module("conv4", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(64, 80, 1).bias(false)));
            bn4 = register_module("bn4", torch::nn::BatchNorm2d(80));
            conv5 = register_module("conv5", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(80, 192, 3).padding(1).bias(false)));
            bn5 = register_module("bn5", torch::nn::BatchNorm2d(192));
            pool2 = register_module("pool2", torch::nn::MaxPool2d(
                                        torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

            // Inception modules
            inception_a1 = register_module("inception_a1", InceptionAModule(192, 32));
            inception_a2 = register_module("inception_a2", InceptionAModule(256, 64));
            inception_a3 = register_module("inception_a3", InceptionAModule(288, 64));
            inception_b = register_module("inception_b", InceptionBModule(288));
            inception_c1 = register_module("inception_c1", InceptionAModule(768, 128));
            inception_c2 = register_module("inception_c2", InceptionAModule(768, 128));
            inception_c3 = register_module("inception_c3", InceptionAModule(768, 128));
            inception_c4 = register_module("inception_c4", InceptionAModule(768, 128));
            inception_d = register_module("inception_d", InceptionBModule(768));
            inception_e1 = register_module("inception_e1", InceptionCModule(1280, 192));
            inception_e2 = register_module("inception_e2", InceptionCModule(2048, 320));

            // Auxiliary classifier
            if (aux_logits_)
            {
                aux_classifier = register_module("aux_classifier", AuxClassifier(768, num_classes));
            }

            // Head
            avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
                                           torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
            dropout = register_module("dropout", torch::nn::Dropout(0.5));
            fc = register_module("fc", torch::nn::Linear(2048, num_classes));
        }

        std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x)
        {
            // Stem: [batch, 3, 32, 32]
            x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 32, 32, 32]
            x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 32, 32, 32]
            x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 64, 32, 32]
            x = pool1->forward(x); // [batch, 64, 16, 16]
            x = torch::relu(bn4->forward(conv4->forward(x))); // [batch, 80, 16, 16]
            x = torch::relu(bn5->forward(conv5->forward(x))); // [batch, 192, 16, 16]
            x = pool2->forward(x); // [batch, 192, 8, 8]

            // Inception-A
            x = inception_a1->forward(x); // [batch, 256, 8, 8]
            x = inception_a2->forward(x); // [batch, 288, 8, 8]
            x = inception_a3->forward(x); // [batch, 288, 8, 8]

            // Inception-B
            x = inception_b->forward(x); // [batch, 768, 4, 4]

            // Inception-C
            x = inception_c1->forward(x); // [batch, 768, 4, 4]
            x = inception_c2->forward(x); // [batch, 768, 4, 4]
            x = inception_c3->forward(x); // [batch, 768, 4, 4]
            torch::Tensor aux_output;
            if (aux_logits_ && training())
            {
                aux_output = aux_classifier->forward(x);
            }
            x = inception_c4->forward(x); // [batch, 768, 4, 4]

            // Inception-D
            x = inception_d->forward(x); // [batch, 1280, 2, 2]

            // Inception-E
            x = inception_e1->forward(x); // [batch, 2048, 2, 2]
            x = inception_e2->forward(x); // [batch, 2048, 2, 2]

            // Head
            x = avg_pool->forward(x); // [batch, 2048, 1, 1]
            x = x.view({x.size(0), -1}); // [batch, 2048]
            x = dropout->forward(x);
            x = fc->forward(x); // [batch, num_classes]

            return std::make_tuple(x, aux_output);
        }

        bool aux_logits_;
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr}, conv5{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
        torch::nn::MaxPool2d pool1{nullptr}, pool2{nullptr};
        torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc{nullptr};
        InceptionAModule inception_a1{nullptr}, inception_a2{nullptr}, inception_a3{nullptr};
        InceptionBModule inception_b{nullptr}, inception_d{nullptr};
        InceptionAModule inception_c1{nullptr}, inception_c2{nullptr}, inception_c3{nullptr}, inception_c4{nullptr};
        InceptionCModule inception_e1{nullptr}, inception_e2{nullptr};
        AuxClassifier aux_classifier{nullptr};
    };

    TORCH_MODULE(InceptionV3);

    // InceptionV4
    struct InceptionV4Impl : torch::nn::Module
    {
        InceptionV4Impl(int num_classes = 10)
        {
            // Stem
            conv1 = register_module("conv1", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
            conv2 = register_module("conv2", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(32, 32, 3).padding(1).bias(false)));
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(32));
            conv3 = register_module("conv3", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
            bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
            pool1 = register_module("pool1", torch::nn::MaxPool2d(
                                        torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
            conv4 = register_module("conv4", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
            bn4 = register_module("bn4", torch::nn::BatchNorm2d(96));
            conv5 = register_module("conv5", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(96, 64, 1).bias(false)));
            bn5 = register_module("bn5", torch::nn::BatchNorm2d(64));
            conv6 = register_module("conv6", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(64, 96, 3).padding(1).bias(false)));
            bn6 = register_module("bn6", torch::nn::BatchNorm2d(96));
            conv7 = register_module("conv7", torch::nn::Conv2d(
                                        torch::nn::Conv2dOptions(160, 192, 3).stride(2).bias(false)));
            bn7 = register_module("bn7", torch::nn::BatchNorm2d(192));

            // Inception modules
            inception_a1 = register_module("inception_a1", InceptionAModule(192));
            inception_a2 = register_module("inception_a2", InceptionAModule(384));
            inception_a3 = register_module("inception_a3", InceptionAModule(384));
            inception_a4 = register_module("inception_a4", InceptionAModule(384));
            reduction_a = register_module("reduction_a", ReductionAModule(384, 192, 224, 256, 384));
            inception_b1 = register_module("inception_b1", InceptionBModule(1024));
            inception_b2 = register_module("inception_b2", InceptionBModule(1024));
            inception_b3 = register_module("inception_b3", InceptionBModule(1024));
            inception_b4 = register_module("inception_b4", InceptionBModule(1024));
            inception_b5 = register_module("inception_b5", InceptionBModule(1024));
            inception_b6 = register_module("inception_b6", InceptionBModule(1024));
            inception_b7 = register_module("inception_b7", InceptionBModule(1024));
            reduction_b = register_module("reduction_b", ReductionBModule(1024));
            inception_c1 = register_module("inception_c1", InceptionCModule(1536));
            inception_c2 = register_module("inception_c2", InceptionCModule(1536));
            inception_c3 = register_module("inception_c3", InceptionCModule(1536));

            // Head
            avg_pool = register_module("avg_pool", torch::nn::AdaptiveAvgPool2d(
                                           torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
            dropout = register_module("dropout", torch::nn::Dropout(0.8));
            fc = register_module("fc", torch::nn::Linear(1536, num_classes));
        }

        torch::Tensor forward(torch::Tensor x)
        {
            // Stem: [batch, 3, 32, 32]
            x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 32, 32, 32]
            x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 32, 32, 32]
            x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 64, 32, 32]
            x = pool1->forward(x); // [batch, 64, 16, 16]
            x = torch::relu(bn4->forward(conv4->forward(x))); // [batch, 96, 16, 16]
            auto branch1 = torch::relu(bn5->forward(conv5->forward(x))); // [batch, 64, 16, 16]
            auto branch2 = torch::relu(bn6->forward(conv6->forward(x))); // [batch, 96, 16, 16]
            x = torch::cat({branch1, branch2}, 1); // [batch, 160, 16, 16]
            x = torch::relu(bn7->forward(conv7->forward(x))); // [batch, 192, 8, 8]

            // Inception-A
            x = inception_a1->forward(x); // [batch, 384, 8, 8]
            x = inception_a2->forward(x); // [batch, 384, 8, 8]
            x = inception_a3->forward(x); // [batch, 384, 8, 8]
            x = inception_a4->forward(x); // [batch, 384, 8, 8]

            // Reduction-A
            x = reduction_a->forward(x); // [batch, 1024, 4, 4]

            // Inception-B
            x = inception_b1->forward(x); // [batch, 1024, 4, 4]
            x = inception_b2->forward(x); // [batch, 1024, 4, 4]
            x = inception_b3->forward(x); // [batch, 1024, 4, 4]
            x = inception_b4->forward(x); // [batch, 1024, 4, 4]
            x = inception_b5->forward(x); // [batch, 1024, 4, 4]
            x = inception_b6->forward(x); // [batch, 1024, 4, 4]
            x = inception_b7->forward(x); // [batch, 1024, 4, 4]

            // Reduction-B
            x = reduction_b->forward(x); // [batch, 1536, 2, 2]

            // Inception-C
            x = inception_c1->forward(x); // [batch, 1536, 2, 2]
            x = inception_c2->forward(x); // [batch, 1536, 2, 2]
            x = inception_c3->forward(x); // [batch, 1536, 2, 2]

            // Head
            x = avg_pool->forward(x); // [batch, 1536, 1, 1]
            x = x.view({x.size(0), -1}); // [batch, 1536]
            x = dropout->forward(x);
            x = fc->forward(x); // [batch, num_classes]
            return x;
        }

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
        torch::nn::Conv2d conv5{nullptr}, conv6{nullptr}, conv7{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};
        torch::nn::BatchNorm2d bn5{nullptr}, bn6{nullptr}, bn7{nullptr};
        torch::nn::MaxPool2d pool1{nullptr};
        torch::nn::AdaptiveAvgPool2d avg_pool{nullptr};
        torch::nn::Dropout dropout{nullptr};
        torch::nn::Linear fc{nullptr};
        InceptionAModule inception_a1{nullptr}, inception_a2{nullptr}, inception_a3{nullptr}, inception_a4{nullptr};
        ReductionAModule reduction_a{nullptr};
        InceptionBModule inception_b1{nullptr}, inception_b2{nullptr}, inception_b3{nullptr};
        InceptionBModule inception_b4{nullptr}, inception_b5{nullptr}, inception_b6{nullptr}, inception_b7{nullptr};
        ReductionBModule reduction_b{nullptr};
        InceptionCModule inception_c1{nullptr}, inception_c2{nullptr}, inception_c3{nullptr};
    };

    TORCH_MODULE(InceptionV4);


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
