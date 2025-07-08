#pragma once

#include "../../common.h"



using namespace std;


namespace xt::models
{

     // Swish activation (x * sigmoid(x))
 torch::Tensor swish(torch::Tensor x) {
     return x * torch::sigmoid(x);
 }

 // Squeeze-and-Excitation Block
 struct SEBlockImpl : torch::nn::Module {
     SEBlockImpl(int in_channels, int reduction) {
         fc1 = register_module("fc1", torch::nn::Linear(in_channels, in_channels / reduction));
         fc2 = register_module("fc2", torch::nn::Linear(in_channels / reduction, in_channels));
     }

     torch::Tensor forward(torch::Tensor x) {
         auto batch = x.size(0);
         auto channels = x.size(1);
         auto avg = torch::avg_pool2d(x, {x.size(2), x.size(3)}).view({batch, channels});
         auto out = torch::relu(fc1->forward(avg));
         out = torch::sigmoid(fc2->forward(out)).view({batch, channels, 1, 1});
         return x * out;
     }

     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
 };
 TORCH_MODULE(SEBlock);

 // MBConv Block (Inverted Residual with Depthwise Separable Conv)
 struct MBConvBlockImpl : torch::nn::Module {
     MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction) {
         int expanded_channels = in_channels * expansion;
         bool has_se = reduction > 0;

         if (expansion != 1) {
             expand_conv = register_module("expand_conv", torch::nn::Conv2d(
                 torch::nn::Conv2dOptions(in_channels, expanded_channels, 1).bias(false)));
             bn0 = register_module("bn0", torch::nn::BatchNorm2d(expanded_channels));
         }

         depthwise_conv = register_module("depthwise_conv", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(expanded_channels, expanded_channels, kernel_size)
                 .stride(stride).padding(kernel_size / 2).groups(expanded_channels).bias(false)));
         bn1 = register_module("bn1", torch::nn::BatchNorm2d(expanded_channels));

         if (has_se) {
             se = register_module("se", SEBlock(expanded_channels, reduction));
         }

         pointwise_conv = register_module("pointwise_conv", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(expanded_channels, out_channels, 1).bias(false)));
         bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));

         skip_connection = (in_channels == out_channels && stride == 1);
     }

     torch::Tensor forward(torch::Tensor x) {
         auto out = x;
         if (expand_conv) {
             out = swish(bn0->forward(expand_conv->forward(out)));
         }
         out = swish(bn1->forward(depthwise_conv->forward(out)));
         if (se) {
             out = se->forward(out);
         }
         out = bn2->forward(pointwise_conv->forward(out));
         if (skip_connection) {
             out += x; // Residual connection
         }
         return out;
     }

     torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
     SEBlock se{nullptr};
     bool skip_connection;
 };
 TORCH_MODULE(MBConvBlock);

 // EfficientNetB0
 struct EfficientNetB0Impl : torch::nn::Module {
     EfficientNetB0Impl(int num_classes = 10) {
         // Initial stem
         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false))); // Simplified stride
         bn0 = register_module("bn0", torch::nn::BatchNorm2d(32));

         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
             {1, 32, 16, 1, 3, 1, 4},   // Stage 1
             {2, 16, 24, 6, 3, 2, 4},   // Stage 2
             {2, 24, 40, 6, 5, 2, 4},   // Stage 3
             {3, 40, 80, 6, 3, 2, 4},   // Stage 4
             {3, 80, 112, 6, 5, 1, 4},  // Stage 5
             {4, 112, 192, 6, 5, 2, 4}, // Stage 6
             {1, 192, 320, 6, 3, 1, 4}  // Stage 7
         };

         int stage_idx = 0;
         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
             for (int i = 0; i < num_repeats; ++i) {
                 int s = (i == 0) ? stride : 1;
                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
                 in_ch = out_ch;
             }
             stage_idx++;
         }

         // Head
         head_conv = register_module("head_conv", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(320, 1280, 1).bias(false)));
         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1280));
         fc = register_module("fc", torch::nn::Linear(1280, num_classes));
     }

     torch::Tensor forward(torch::Tensor x) {
         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 32, 32, 32]
         for (auto& block : *blocks) {
             x = block->forward(x);
         }
         x = swish(bn1->forward(head_conv->forward(x)));
         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1280, 1, 1]
         x = x.view({x.size(0), -1}); // [batch, 1280]
         x = fc->forward(x); // [batch, num_classes]
         return x;
     }

     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
     torch::nn::Linear fc{nullptr};
     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
 };
 TORCH_MODULE(EfficientNetB0);






    struct EfficientNetB0 : xt::Cloneable<EfficientNetB0>
    {
    private:

    public:
        EfficientNetB0(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB0(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };


    struct EfficientNetB1 : xt::Cloneable<EfficientNetB1>
    {
    private:

    public:
        EfficientNetB1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB1(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };


    struct EfficientNetB2 : xt::Cloneable<EfficientNetB2>
    {
    private:

    public:
        EfficientNetB2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB2(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };

    struct EfficientNetB3 : xt::Cloneable<EfficientNetB3>
    {
    private:

    public:
        EfficientNetB3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB3(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };

    struct EfficientNetB4 : xt::Cloneable<EfficientNetB4>
    {
    private:

    public:
        EfficientNetB4(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB4(int num_classes, int in_channels, std::vector<int64_t> input_shape);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };

    struct EfficientNetB5 : xt::Cloneable<EfficientNetB5>
    {
    private:

    public:
        EfficientNetB5(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB5(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };

    struct EfficientNetB6 : xt::Cloneable<EfficientNetB6>
    {
    private:

    public:
        EfficientNetB6(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB6(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

        void reset() override;
    };

    struct EfficientNetB7 : xt::Cloneable<EfficientNetB7>
    {
    private:

    public:
        EfficientNetB7(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB7(int num_classes, int in_channels, std::vector<int64_t> input_shape);
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}
