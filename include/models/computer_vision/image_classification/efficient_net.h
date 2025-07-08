#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models {

    // Swish activation (x * sigmoid(x))
    torch::Tensor swish(torch::Tensor x);

    // Squeeze-and-Excitation Block
    struct SEBlockImpl : torch::nn::Module {
        SEBlockImpl(int in_channels, int reduction);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };

    TORCH_MODULE(SEBlock);

    // MBConv Block (Inverted Residual with Depthwise Separable Conv)
    struct MBConvBlockImpl : torch::nn::Module {
        MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
        SEBlock se{nullptr};
        bool skip_connection;
    };

    TORCH_MODULE(MBConvBlock);

    // EfficientNetB0
    struct EfficientNetB0Impl : torch::nn::Module {
        EfficientNetB0Impl(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
        torch::nn::Linear fc{nullptr};
        torch::nn::ModuleList blocks{torch::nn::ModuleList()};
    };

    TORCH_MODULE(EfficientNetB0);


    struct EfficientNetB0 : xt::Cloneable<EfficientNetB0> {
    private:

    public:
        EfficientNetB0(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB0(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // EfficientNetB1
    struct EfficientNetB1Impl : torch::nn::Module {
        EfficientNetB1Impl(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
        torch::nn::Linear fc{nullptr};
        torch::nn::ModuleList blocks{torch::nn::ModuleList()};
    };

    TORCH_MODULE(EfficientNetB1);


    struct EfficientNetB1 : xt::Cloneable<EfficientNetB1> {
    private:

    public:
        EfficientNetB1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB1(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


     // EfficientNetB2
 struct EfficientNetB2Impl : torch::nn::Module {
     EfficientNetB2Impl(int num_classes = 10) {
         // Initial stem
         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
             torch::nn::Conv2dOptions(3, 32, 3).stride(1).padding(1).bias(false))); // Simplified stride
         bn0 = register_module("bn0", torch::nn::BatchNorm2d(32));

         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
             {2, 32, 16, 1, 3, 1, 4},   // Stage 1 (increased repeats vs. B1)
             {3, 16, 24, 6, 3, 2, 4},   // Stage 2 (increased repeats vs. B1)
             {3, 24, 48, 6, 5, 2, 4},   // Stage 3 (increased out_channels vs. B1)
             {4, 48, 88, 6, 3, 2, 4},   // Stage 4 (increased repeats and out_channels vs. B1)
             {4, 88, 120, 6, 5, 1, 4},  // Stage 5 (increased out_channels vs. B1)
             {5, 120, 208, 6, 5, 2, 4}, // Stage 6 (increased repeats and out_channels vs. B1)
             {2, 208, 352, 6, 3, 1, 4}, // Stage 7 (increased out_channels vs. B1)
             {1, 352, 352, 6, 3, 1, 4}  // Stage 8
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
             torch::nn::Conv2dOptions(352, 1408, 1).bias(false))); // Increased channels vs. B1
         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1408));
         fc = register_module("fc", torch::nn::Linear(1408, num_classes));
     }

     torch::Tensor forward(torch::Tensor x) {
         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 32, 32, 32]
         for (auto& block : *blocks) {
             x = block->forward(x);
         }
         x = swish(bn1->forward(head_conv->forward(x)));
         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1408, 1, 1]
         x = x.view({x.size(0), -1}); // [batch, 1408]
         x = fc->forward(x); // [batch, num_classes]
         return x;
     }

     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
     torch::nn::Linear fc{nullptr};
     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
 };
 TORCH_MODULE(EfficientNetB2);


    struct EfficientNetB2 : xt::Cloneable<EfficientNetB2> {
    private:

    public:
        EfficientNetB2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB2(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // // EfficientNetB3
// struct EfficientNetB3Impl : torch::nn::Module {
//     EfficientNetB3Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 40, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(40));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {2, 40, 24, 1, 3, 1, 4},   // Stage 1
//             {3, 24, 32, 6, 3, 2, 4},   // Stage 2
//             {4, 32, 48, 6, 5, 2, 4},   // Stage 3
//             {4, 48, 96, 6, 3, 2, 4},   // Stage 4
//             {5, 96, 136, 6, 5, 1, 4},  // Stage 5
//             {6, 136, 232, 6, 5, 2, 4}, // Stage 6
//             {3, 232, 384, 6, 3, 1, 4}, // Stage 7
//             {1, 384, 384, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(384, 1536, 1).bias(false))); // Increased channels vs. B2
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1536));
//         fc = register_module("fc", torch::nn::Linear(1536, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 40, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1536, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1536]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB3);



    struct EfficientNetB3 : xt::Cloneable<EfficientNetB3> {
    private:

    public:
        EfficientNetB3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB3(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };



    // // EfficientNetB4
// struct EfficientNetB4Impl : torch::nn::Module {
//     EfficientNetB4Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 48, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(48));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {2, 48, 24, 1, 3, 1, 4},   // Stage 1
//             {4, 24, 32, 6, 3, 2, 4},   // Stage 2
//             {4, 32, 56, 6, 5, 2, 4},   // Stage 3
//             {6, 56, 112, 6, 3, 2, 4},  // Stage 4
//             {6, 112, 160, 6, 5, 1, 4}, // Stage 5
//             {8, 160, 272, 6, 5, 2, 4}, // Stage 6
//             {3, 272, 448, 6, 3, 1, 4}, // Stage 7
//             {1, 448, 448, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(448, 1792, 1).bias(false))); // Increased channels vs. B3
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(1792));
//         fc = register_module("fc", torch::nn::Linear(1792, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 48, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 1792, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 1792]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB4);

    struct EfficientNetB4 : xt::Cloneable<EfficientNetB4> {
    private:

    public:
        EfficientNetB4(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB4(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // // EfficientNetB5
// struct EfficientNetB5Impl : torch::nn::Module {
//     EfficientNetB5Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 48, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(48));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {3, 48, 24, 1, 3, 1, 4},   // Stage 1
//             {5, 24, 40, 6, 3, 2, 4},   // Stage 2
//             {5, 40, 64, 6, 5, 2, 4},   // Stage 3
//             {7, 64, 128, 6, 3, 2, 4},  // Stage 4
//             {8, 128, 176, 6, 5, 1, 4}, // Stage 5
//             {9, 176, 304, 6, 5, 2, 4}, // Stage 6
//             {4, 304, 512, 6, 3, 1, 4}, // Stage 7
//             {2, 512, 512, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(512, 2048, 1).bias(false))); // Increased channels vs. B4
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(2048));
//         fc = register_module("fc", torch::nn::Linear(2048, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 48, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 2048, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 2048]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB5);

    struct EfficientNetB5 : xt::Cloneable<EfficientNetB5> {
    private:

    public:
        EfficientNetB5(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB5(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // // EfficientNetB6
// struct EfficientNetB6Impl : torch::nn::Module {
//     EfficientNetB6Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 56, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(56));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {3, 56, 32, 1, 3, 1, 4},   // Stage 1
//             {5, 32, 40, 6, 3, 2, 4},   // Stage 2
//             {6, 40, 72, 6, 5, 2, 4},   // Stage 3
//             {8, 72, 144, 6, 3, 2, 4},  // Stage 4
//             {9, 144, 200, 6, 5, 1, 4}, // Stage 5
//             {11, 200, 344, 6, 5, 2, 4}, // Stage 6
//             {5, 344, 576, 6, 3, 1, 4}, // Stage 7
//             {2, 576, 576, 6, 3, 1, 4}  // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(576, 2304, 1).bias(false))); // Increased channels vs. B5
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(2304));
//         fc = register_module("fc", torch::nn::Linear(2304, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 56, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 2304, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 2304]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB6);
//



    struct EfficientNetB6 : xt::Cloneable<EfficientNetB6> {
    private:

    public:
        EfficientNetB6(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB6(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // // EfficientNetB7
// struct EfficientNetB7Impl : torch::nn::Module {
//     EfficientNetB7Impl(int num_classes = 10) {
//         // Initial stem
//         stem_conv = register_module("stem_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1).bias(false))); // Simplified stride, increased channels
//         bn0 = register_module("bn0", torch::nn::BatchNorm2d(64));
//
//         // MBConv blocks configuration: {num_repeats, in_channels, out_channels, expansion, kernel_size, stride, se_reduction}
//         std::vector<std::tuple<int, int, int, int, int, int, int>> config = {
//             {4, 64, 32, 1, 3, 1, 4},   // Stage 1
//             {6, 32, 48, 6, 3, 2, 4},   // Stage 2
//             {7, 48, 80, 6, 5, 2, 4},   // Stage 3
//             {10, 80, 160, 6, 3, 2, 4}, // Stage 4
//             {11, 160, 224, 6, 5, 1, 4}, // Stage 5
//             {13, 224, 384, 6, 5, 2, 4}, // Stage 6
//             {6, 384, 640, 6, 3, 1, 4},  // Stage 7
//             {2, 640, 640, 6, 3, 1, 4}   // Stage 8
//         };
//
//         int stage_idx = 0;
//         for (const auto& [num_repeats, in_ch, out_ch, expansion, kernel, stride, reduction] : config) {
//             for (int i = 0; i < num_repeats; ++i) {
//                 int s = (i == 0) ? stride : 1;
//                 blocks->push_back(MBConvBlock(in_ch, out_ch, expansion, kernel, s, reduction));
//                 register_module("block_" + std::to_string(stage_idx) + "_" + std::to_string(i), blocks->back());
//                 in_ch = out_ch;
//             }
//             stage_idx++;
//         }
//
//         // Head
//         head_conv = register_module("head_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(640, 2560, 1).bias(false))); // Increased channels vs. B6
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(2560));
//         fc = register_module("fc", torch::nn::Linear(2560, num_classes));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = swish(bn0->forward(stem_conv->forward(x))); // [batch, 64, 32, 32]
//         for (auto& block : *blocks) {
//             x = block->forward(x);
//         }
//         x = swish(bn1->forward(head_conv->forward(x)));
//         x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, 2560, 1, 1]
//         x = x.view({x.size(0), -1}); // [batch, 2560]
//         x = fc->forward(x); // [batch, num_classes]
//         return x;
//     }
//
//     torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
//     torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
//     torch::nn::Linear fc{nullptr};
//     torch::nn::ModuleList blocks{torch::nn::ModuleList()};
// };
// TORCH_MODULE(EfficientNetB7);

    struct EfficientNetB7 : xt::Cloneable<EfficientNetB7> {
    private:

    public:
        EfficientNetB7(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB7(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };

}
