#pragma once

#include "../../common.h"


namespace xt::models {
    // --- The ZefNet Model, Adapted for MNIST ---
    // This model follows the spirit of ZefNet (multiple conv stages with LRN)
    // but is adapted for the smaller 28x28 MNIST input.

    struct ZefNet : xt::Module {
        // We separate the model into a feature extractor and a classifier
        torch::nn::Sequential features, classifier;

        ZefNet(int num_classes = 10);
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);
    };

    // TORCH_MODULE(ZefNet);

//    struct ZefNet : xt::Cloneable<ZefNet>
//    {
//    private:
//
//    public:
//        ZefNet(int num_classes /* classes */, int in_channels = 3/* input channels */);
//
//        ZefNet(int num_classes, int in_channels, std::vector<int64_t> input_shape);
//
//        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
//        void reset() override;
//    };
}
