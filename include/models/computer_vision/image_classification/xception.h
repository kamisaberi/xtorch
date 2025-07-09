#pragma once
#include "../../common.h"

namespace xt::models
{
     // --- The Core Building Block: Depthwise Separable Convolution ---
     struct SeparableConv2dImpl : torch::nn::Module {
         torch::nn::Conv2d depthwise, pointwise;

         SeparableConv2dImpl(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
             // Depthwise convolution: processes each channel independently
             : depthwise(torch::nn::Conv2dOptions(in_channels, in_channels, kernel_size)
                             .stride(stride).padding(padding).groups(in_channels).bias(false)),
               // Pointwise convolution: 1x1 conv to mix channel information
               pointwise(torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false))
         {
             register_module("depthwise", depthwise);
             register_module("pointwise", pointwise);
         }

         torch::Tensor forward(torch::Tensor x) {
             return pointwise(depthwise(x));
         }
     };
     TORCH_MODULE(SeparableConv2d);


     // --- The Main Repeating Block in Xception ---
     struct XceptionBlockImpl : torch::nn::Module {
         torch::nn::Sequential block, shortcut;

         XceptionBlockImpl(int in_channels, int out_channels, int num_reps, int stride, bool start_with_relu = true) {
             torch::nn::Sequential layers;
             if (start_with_relu) {
                 layers->push_back(torch::nn::ReLU());
             }

             // Add the separable convolutions
             layers->push_back(SeparableConv2d(in_channels, out_channels, 3, 1, 1));
             layers->push_back(torch::nn::BatchNorm2d(out_channels));
             for (int i = 1; i < num_reps; ++i) {
                 layers->push_back(torch::nn::ReLU());
                 layers->push_back(SeparableConv2d(out_channels, out_channels, 3, 1, 1));
                 layers->push_back(torch::nn::BatchNorm2d(out_channels));
             }

             // The final conv in the block may have a stride
             if (stride != 1) {
                 layers->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(stride).padding(1)));
             }
             block = register_module("block", layers);

             // Shortcut connection to match dimensions if they change
             if (stride != 1 || in_channels != out_channels) {
                 shortcut = register_module("shortcut", torch::nn::Sequential(
                     torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1).stride(stride).bias(false)),
                     torch::nn::BatchNorm2d(out_channels)
                 ));
             }
         }

         torch::Tensor forward(torch::Tensor x) {
             auto out = block->forward(x);
             auto short_x = shortcut ? shortcut->forward(x) : x;
             return out + short_x;
         }
     };
     TORCH_MODULE(XceptionBlock);


     // --- The Full Xception Model ---
     struct XceptionImpl : torch::nn::Module {
         torch::nn::Conv2d conv1, conv2;
         torch::nn::BatchNorm2d bn1, bn2;
         torch::nn::Sequential entry_flow, middle_flow, exit_flow;
         torch::nn::Linear fc;

         XceptionImpl(int num_middle_blocks, int num_classes = 10) {
             // --- Entry Flow ---
             conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).stride(2).padding(1).bias(false)));
             bn1 = register_module("bn1", torch::nn::BatchNorm2d(32));
             conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1).bias(false)));
             bn2 = register_module("bn2", torch::nn::BatchNorm2d(64));

             entry_flow = torch::nn::Sequential(
                 XceptionBlock(64, 128, 2, 2, false),
                 XceptionBlock(128, 256, 2, 2),
                 XceptionBlock(256, 728, 2, 2)
             );
             register_module("entry_flow", entry_flow);

             // --- Middle Flow ---
             middle_flow = torch::nn::Sequential();
             for (int i=0; i < num_middle_blocks; ++i) {
                 middle_flow->push_back(XceptionBlock(728, 728, 3, 1));
             }
             register_module("middle_flow", middle_flow);

             // --- Exit Flow ---
             exit_flow = torch::nn::Sequential(
                 XceptionBlock(728, 1024, 2, 2),
                 SeparableConv2d(1024, 1536, 3, 1, 1),
                 torch::nn::BatchNorm2d(1536),
                 torch::nn::ReLU(),
                 SeparableConv2d(1536, 2048, 3, 1, 1),
                 torch::nn::BatchNorm2d(2048),
                 torch::nn::ReLU()
             );
             register_module("exit_flow", exit_flow);

             // --- Classifier ---
             fc = register_module("fc", torch::nn::Linear(2048, num_classes));
         }

         torch::Tensor forward(torch::Tensor x) {
             x = torch::relu(bn1(conv1(x)));
             x = torch::relu(bn2(conv2(x)));
             x = entry_flow->forward(x);
             x = middle_flow->forward(x);
             x = exit_flow->forward(x);

             // Global Average Pooling
             x = torch::nn::functional::adaptive_avg_pool2d(x, torch::nn::functional::AdaptiveAvgPool2dFuncOptions(1));
             x = x.view({x.size(0), -1});
             x = fc->forward(x);
             return x;
         }
     };
     TORCH_MODULE(Xception);

//    struct Xception : xt::Cloneable<Xception>
//    {
//    private:
//
//    public:
//        Xception(int num_classes /* classes */, int in_channels = 3/* input channels */);
//
//        Xception(int num_classes, int in_channels, std::vector<int64_t> input_shape);
//
//        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//        void reset() override;
//    };
}
