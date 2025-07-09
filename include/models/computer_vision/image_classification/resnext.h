#pragma once
#include "../../common.h"


using namespace std;

namespace xt::models
{
     // --- The Core ResNeXt Block ---

     struct ResNeXtBlockImpl : torch::nn::Module {
         // A ResNeXt block splits the input into 'cardinality' groups,
         // processes each with a small bottleneck, and then merges them.
         // This is implemented efficiently using a grouped convolution.
         torch::nn::Conv2d conv1, conv2_grouped, conv3;
         torch::nn::BatchNorm2d bn1, bn2, bn3;

         // Shortcut connection to handle dimension changes
         torch::nn::Sequential shortcut;

         ResNeXtBlockImpl(int in_channels, int out_channels, int stride, int cardinality, int bottleneck_width) {
             int group_channels = cardinality * bottleneck_width;

             // 1x1 convolution to enter the bottleneck
             conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, group_channels, 1).bias(false)));
             bn1 = register_module("bn1", torch::nn::BatchNorm2d(group_channels));

             // The core grouped convolution. This is the "split-transform" part.
             // It has `cardinality` groups, each with `bottleneck_width` input/output channels.
             conv2_grouped = register_module("conv2", torch::nn::Conv2d(
                 torch::nn::Conv2dOptions(group_channels, group_channels, 3)
                     .stride(stride)
                     .padding(1)
                     .groups(cardinality) // The key parameter for ResNeXt
                     .bias(false)
             ));
             bn2 = register_module("bn2", torch::nn::BatchNorm2d(group_channels));

             // 1x1 convolution to exit the bottleneck and project to the final output channels
             conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(group_channels, out_channels, 1).bias(false)));
             bn3 = register_module("bn3", torch::nn::BatchNorm2d(out_channels));

             // If dimensions change (stride > 1 or in_channels != out_channels),
             // we need to project the shortcut connection.
             if (stride != 1 || in_channels != out_channels) {
                 shortcut = torch::nn::Sequential(
                     torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
                     torch::nn::BatchNorm2d(out_channels)
                 );
             }
             register_module("shortcut", shortcut);
         }

         torch::Tensor forward(torch::Tensor x) {
             auto out = torch::relu(bn1(conv1(x)));
             out = torch::relu(bn2(conv2_grouped(out)));
             out = bn3(conv3(out));

             // Apply shortcut, either identity or projection
             out += shortcut ? shortcut->forward(x) : x;
             out = torch::relu(out);
             return out;
         }
     };
     TORCH_MODULE(ResNeXtBlock);


     // --- The Full ResNeXt Model ---

     struct ResNeXtImpl : torch::nn::Module {
         torch::nn::Conv2d conv1;
         torch::nn::BatchNorm2d bn1;
         torch::nn::Sequential layer1, layer2, layer3;
         torch::nn::Linear linear;

         ResNeXtImpl(int num_blocks1, int num_blocks2, int num_blocks3, int cardinality, int bottleneck_width, int num_classes = 10) {
             // --- Stem for MNIST ---
             conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).stride(1).padding(1).bias(false)));
             bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));

             int in_channels = 64;

             // --- Stage 1 ---
             layer1 = torch::nn::Sequential();
             int out_channels = cardinality * bottleneck_width * 2;
             layer1->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width));
             in_channels = out_channels;
             for (int i = 1; i < num_blocks1; ++i) {
                 layer1->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width));
             }
             register_module("layer1", layer1);

             // --- Stage 2 ---
             layer2 = torch::nn::Sequential();
             out_channels = cardinality * bottleneck_width * 4;
             layer2->push_back(ResNeXtBlock(in_channels, out_channels, 2, cardinality, bottleneck_width * 2)); // Downsample
             in_channels = out_channels;
             for (int i = 1; i < num_blocks2; ++i) {
                 layer2->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width * 2));
             }
             register_module("layer2", layer2);

             // --- Stage 3 ---
             layer3 = torch::nn::Sequential();
             out_channels = cardinality * bottleneck_width * 8;
             layer3->push_back(ResNeXtBlock(in_channels, out_channels, 2, cardinality, bottleneck_width * 4)); // Downsample
             in_channels = out_channels;
             for (int i = 1; i < num_blocks3; ++i) {
                 layer3->push_back(ResNeXtBlock(in_channels, out_channels, 1, cardinality, bottleneck_width * 4));
             }
             register_module("layer3", layer3);

             // --- Classifier ---
             linear = register_module("linear", torch::nn::Linear(out_channels, num_classes));
         }

         torch::Tensor forward(torch::Tensor x) {
             x = torch::relu(bn1(conv1(x)));
             x = layer1->forward(x);
             x = layer2->forward(x);
             x = layer3->forward(x);

             // Global Average Pooling
             x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));

             x = x.view({x.size(0), -1});
             x = linear->forward(x);
             return x;
         }
     };
     TORCH_MODULE(ResNeXt);


//    struct ResNeXt : xt::Cloneable<ResNeXt>
//    {
//    private:
//
//    public:
//        ResNeXt(int num_classes /* classes */, int in_channels = 3/* input channels */);
//
//        ResNeXt(int num_classes, int in_channels, std::vector<int64_t> input_shape);
//
//        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//        void reset() override;
//    };
}
