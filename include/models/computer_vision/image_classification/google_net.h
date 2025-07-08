#pragma once
#include "../../common.h"





namespace xt::models
{


     // Inception Module
 struct InceptionModuleImpl : torch::nn::Module {
     InceptionModuleImpl(int in_channels, int ch1x1, int ch3x3red, int ch3x3, int ch5x5red, int ch5x5, int pool_proj) {
         // Branch 1: 1x1 conv
         conv1x1 = register_module("conv1x1", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(in_channels, ch1x1, 1).stride(1)));
         bn1x1 = register_module("bn1x1", torch::nn::BatchNorm2d(ch1x1));

         // Branch 2: 1x1 conv -> 3x3 conv
         conv3x3_reduce = register_module("conv3x3_reduce", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(in_channels, ch3x3red, 1).stride(1)));
         bn3x3_reduce = register_module("bn3x3_reduce", torch::nn::BatchNorm2d(ch3x3red));
         conv3x3 = register_module("conv3x3", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(ch3x3red, ch3x3, 3).stride(1).padding(1)));
         bn3x3 = register_module("bn3x3", torch::nn::BatchNorm2d(ch3x3));

         // Branch 3: 1x1 conv -> 5x5 conv
         conv5x5_reduce = register_module("conv5x5_reduce", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(in_channels, ch5x5red, 1).stride(1)));
         bn5x5_reduce = register_module("bn5x5_reduce", torch::nn::BatchNorm2d(ch5x5red));
         conv5x5 = register_module("conv5x5", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(ch5x5red, ch5x5, 5).stride(1).padding(2)));
         bn5x5 = register_module("bn5x5", torch::nn::BatchNorm2d(ch5x5));

         // Branch 4: 3x3 max pool -> 1x1 conv
         pool = register_module("pool", torch::nn::MaxPool2d(
             torch::nn::MaxPool2dOptions(3).stride(1).padding(1)));
         pool_proj = register_module("pool_proj", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(in_channels, pool_proj, 1).stride(1)));
         bn_pool_proj = register_module("bn_pool_proj", torch::nn::BatchNorm2d(pool_proj));
     }

     torch::Tensor forward(torch::Tensor x) {
         // x: [batch, in_channels, h, w]
         // Branch 1
         auto b1 = torch::relu(bn1x1->forward(conv1x1->forward(x))); // [batch, ch1x1, h, w]

         // Branch 2
         auto b2 = torch::relu(bn3x3_reduce->forward(conv3x3_reduce->forward(x)));
         b2 = torch::relu(bn3x3->forward(conv3x3->forward(b2))); // [batch, ch3x3, h, w]

         // Branch 3
         auto b3 = torch::relu(bn5x5_reduce->forward(conv5x5_reduce->forward(x)));
         b3 = torch::relu(bn5x5->forward(conv5x5->forward(b3))); // [batch, ch5x5, h, w]

         // Branch 4
         auto b4 = pool->forward(x);
         b4 = torch::relu(bn_pool_proj->forward(pool_proj->forward(b4))); // [batch, pool_proj, h, w]

         // Concatenate along channel dimension
         return torch::cat({b1, b2, b3, b4}, 1); // [batch, ch1x1+ch3x3+ch5x5+pool_proj, h, w]
     }

     torch::nn::Conv2d conv1x1{nullptr}, conv3x3_reduce{nullptr}, conv3x3{nullptr};
     torch::nn::Conv2d conv5x5_reduce{nullptr}, conv5x5{nullptr}, pool_proj{nullptr};
     torch::nn::BatchNorm2d bn1x1{nullptr}, bn3x3_reduce{nullptr}, bn3x3{nullptr};
     torch::nn::BatchNorm2d bn5x5_reduce{nullptr}, bn5x5{nullptr}, bn_pool_proj{nullptr};
     torch::nn::MaxPool2d pool{nullptr};
 };
 TORCH_MODULE(InceptionModule);

 // Simplified GoogLeNet
 struct GoogLeNetImpl : torch::nn::Module {
     GoogLeNetImpl(int in_channels, int num_classes) {
         // Stem
         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)));
         stem_bn = register_module("stem_bn", torch::nn::BatchNorm2d(64));

         // Inception blocks
         inception1 = register_module("inception1", InceptionModule(64, 32, 16, 32, 8, 16, 16)); // Output: 96
         inception2 = register_module("inception2", InceptionModule(96, 48, 24, 48, 12, 24, 24)); // Output: 144

         // Downsampling
         pool = register_module("pool", torch::nn::MaxPool2d(
             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));

         // Classifier
         global_pool = register_module("global_pool", torch::nn::AdaptiveAvgPool2d(1));
         fc = register_module("fc", torch::nn::Linear(144, num_classes));
     }

     torch::Tensor forward(torch::Tensor x) {
         // x: [batch, in_channels, 32, 32]
         x = torch::relu(stem_bn->forward(stem_conv->forward(x))); // [batch, 64, 32, 32]
         x = inception1->forward(x); // [batch, 96, 32, 32]
         x = pool->forward(x); // [batch, 96, 16, 16]
         x = inception2->forward(x); // [batch, 144, 16, 16]
         x = global_pool->forward(x).view({x.size(0), -1}); // [batch, 144]
         x = fc->forward(x); // [batch, num_classes]
         return x;
     }

     torch::nn::Conv2d stem_conv{nullptr};
     torch::nn::BatchNorm2d stem_bn{nullptr};
     InceptionModule inception1{nullptr}, inception2{nullptr};
     torch::nn::MaxPool2d pool{nullptr};
     torch::nn::AdaptiveAvgPool2d global_pool{nullptr};
     torch::nn::Linear fc{nullptr};
 };
 TORCH_MODULE(GoogLeNet);






    struct GoogLeNet : xt::Cloneable<GoogLeNet>
    {
    private:

    public:
        GoogLeNet(int num_classes /* classes */, int in_channels = 3/* input channels */);

        GoogLeNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };


}
