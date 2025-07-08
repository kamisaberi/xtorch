#pragma once
#include "../../common.h"


namespace xt::models
{
    // // --- The ZefNet Model, Adapted for MNIST ---
    // // This model follows the spirit of ZefNet (multiple conv stages with LRN)
    // // but is adapted for the smaller 28x28 MNIST input.
    //
    // struct ZefNetImpl : torch::nn::Module {
    //     // We separate the model into a feature extractor and a classifier
    //     torch::nn::Sequential features, classifier;
    //
    //     ZefNetImpl(int num_classes = 10) {
    //
    //         // --- Feature Extractor ---
    //         // A sequence of Convolution, ReLU, Pooling, and LRN layers
    //         features = torch::nn::Sequential(
    //             // Stage 1
    //             // In: 1x28x28, Out: 64x14x14
    //             // Original ZefNet used 7x7 stride 2. We use 5x5 stride 1 + MaxPool to be gentler on the small input.
    //             torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 5).stride(1).padding(2)),
    //             torch::nn::ReLU(),
    //             torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
    //             // Local Response Normalization was key to ZefNet/AlexNet
    //             torch::nn::LocalResponseNorm(torch::nn::LocalResponseNormOptions(5).alpha(0.0001).beta(0.75).k(2.0)),
    //
    //             // Stage 2
    //             // In: 64x14x14, Out: 256x7x7
    //             torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 256, 5).stride(1).padding(2)),
    //             torch::nn::ReLU(),
    //             torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)),
    //             torch::nn::LocalResponseNorm(torch::nn::LocalResponseNormOptions(5).alpha(0.0001).beta(0.75).k(2.0)),
    //
    //             // Stage 3
    //             // In: 256x7x7, Out: 384x7x7
    //             torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 384, 3).stride(1).padding(1)),
    //             torch::nn::ReLU(),
    //
    //             // Stage 4
    //             // In: 384x7x7, Out: 384x7x7
    //             torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 384, 3).stride(1).padding(1)),
    //             torch::nn::ReLU(),
    //
    //             // Stage 5
    //             // In: 384x7x7, Out: 256x3x3
    //             torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)),
    //             torch::nn::ReLU(),
    //             torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
    //         );
    //         register_module("features", features);
    //
    //         // --- Classifier ---
    //         // The original 4096-neuron layers are too large. We use a smaller MLP.
    //         // The input size is 256 channels * 3 * 3 spatial dimensions.
    //         classifier = torch::nn::Sequential(
    //             torch::nn::Dropout(0.5),
    //             torch::nn::Linear(256 * 3 * 3, 512),
    //             torch::nn::ReLU(),
    //             torch::nn::Dropout(0.5),
    //             torch::nn::Linear(512, num_classes)
    //         );
    //         register_module("classifier", classifier);
    //     }
    //
    //     torch::Tensor forward(torch::Tensor x) {
    //         x = features->forward(x);
    //         // Flatten the feature maps before the classifier
    //         x = torch::flatten(x, 1);
    //         x = classifier->forward(x);
    //         return x;
    //     }
    // };
    // TORCH_MODULE(ZefNet);

    struct ZefNet : xt::Cloneable<ZefNet>
    {
    private:

    public:
        ZefNet(int num_classes /* classes */, int in_channels = 3/* input channels */);

        ZefNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
