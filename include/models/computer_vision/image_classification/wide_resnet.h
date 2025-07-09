#pragma once
#include "../../common.h"


using namespace std;

namespace xt::models
{
     // --- The Core Wide-ResNet Block ---
     // This block is different from a standard ResNet block. It uses a (BN-ReLU-Conv) pre-activation
     // sequence and includes a Dropout layer.

     struct WideBasicBlockImpl : torch::nn::Module {
         torch::nn::BatchNorm2d bn1, bn2;
         torch::nn::Conv2d conv1, conv2;
         torch::nn::Dropout dropout;

         // Shortcut connection for residual path
         torch::nn::Sequential shortcut;

         WideBasicBlockImpl(int in_planes, int planes, double dropout_rate, int stride = 1)
             : bn1(in_planes),
               conv1(torch::nn::Conv2dOptions(in_planes, planes, 3).stride(stride).padding(1).bias(false)),
               bn2(planes),
               conv2(torch::nn::Conv2dOptions(planes, planes, 3).stride(1).padding(1).bias(false)),
               dropout(dropout_rate)
         {
             register_module("bn1", bn1);
             register_module("conv1", conv1);
             register_module("bn2", bn2);
             register_module("conv2", conv2);
             register_module("dropout", dropout);

             // If dimensions change, we need to project the shortcut with a 1x1 conv
             if (stride != 1 || in_planes != planes) {
                 shortcut = torch::nn::Sequential(
                     torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1).stride(stride).bias(false))
                 );
                 register_module("shortcut", shortcut);
             }
         }

         torch::Tensor forward(torch::Tensor x) {
             auto out = torch::relu(bn1(x));
             out = conv1(out);
             out = dropout(out); // Dropout is applied here in the WideResNet block
             out = torch::relu(bn2(out));
             out = conv2(out);

             auto short_x = shortcut ? shortcut->forward(x) : x;
             out += short_x;

             return out;
         }
     };
     TORCH_MODULE(WideBasicBlock);


     // --- The Full WideResNet Model ---

     struct WideResNetImpl : torch::nn::Module {
         torch::nn::Conv2d conv1;
         torch::nn::Sequential layer1, layer2, layer3;
         torch::nn::BatchNorm2d bn_final;
         torch::nn::Linear linear;

         WideResNetImpl(int depth, int widen_factor, double dropout_rate, int num_classes = 10) {
             // Depth must be of the form 6*N + 4
             assert((depth - 4) % 6 == 0 && "WideResNet depth should be 6n+4");
             int N = (depth - 4) / 6;

             // --- Stem (for MNIST, 1 input channel) ---
             conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1).bias(false)));

             int in_planes = 16;

             // --- Stage 1 ---
             int out_planes1 = 16 * widen_factor;
             layer1 = _make_layer(in_planes, out_planes1, N, 1, dropout_rate);
             in_planes = out_planes1;

             // --- Stage 2 ---
             int out_planes2 = 32 * widen_factor;
             layer2 = _make_layer(in_planes, out_planes2, N, 2, dropout_rate); // Downsample
             in_planes = out_planes2;

             // --- Stage 3 ---
             int out_planes3 = 64 * widen_factor;
             layer3 = _make_layer(in_planes, out_planes3, N, 2, dropout_rate); // Downsample
             in_planes = out_planes3;

             register_module("layer1", layer1);
             register_module("layer2", layer2);
             register_module("layer3", layer3);

             // --- Classifier ---
             bn_final = register_module("bn_final", torch::nn::BatchNorm2d(in_planes));
             linear = register_module("linear", torch::nn::Linear(in_planes, num_classes));
         }

         torch::nn::Sequential _make_layer(int in_planes, int planes, int num_blocks, int stride, double dropout_rate) {
             torch::nn::Sequential layers;
             layers->push_back(WideBasicBlock(in_planes, planes, dropout_rate, stride));
             for (int i = 1; i < num_blocks; ++i) {
                 layers->push_back(WideBasicBlock(planes, planes, dropout_rate, 1));
             }
             return layers;
         }

         torch::Tensor forward(torch::Tensor x) {
             x = conv1(x);
             x = layer1->forward(x);
             x = layer2->forward(x);
             x = layer3->forward(x);

             x = torch::relu(bn_final(x));

             // Global Average Pooling
             x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));

             x = x.view({x.size(0), -1});
             x = linear->forward(x);
             return x;
         }
     };
     TORCH_MODULE(WideResNet);


//    struct WideResNet : xt::Cloneable<WideResNet>
//    {
//    private:
//
//    public:
//        WideResNet(int num_classes /* classes */, int in_channels = 3/* input channels */);
//
//        WideResNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);
//
//        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//        void reset() override;
//    };
}
