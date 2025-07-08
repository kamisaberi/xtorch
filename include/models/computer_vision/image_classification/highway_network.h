#pragma once

#include "../../common.h"

namespace xt::models {

    // Highway Layer
    struct HighwayLayerImpl : torch::nn::Module {
        HighwayLayerImpl(int input_size);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Linear transform{nullptr}, plain{nullptr};
    };

    TORCH_MODULE(HighwayLayer);

    // Highway Network
    struct HighwayNetworkImpl : torch::nn::Module {
        HighwayNetworkImpl(int input_size, int num_classes, int num_layers);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Linear input_layer{nullptr}, output_layer{nullptr};
        torch::nn::ModuleList layers{torch::nn::ModuleList()};
    };

    TORCH_MODULE(HighwayNetwork);


    struct HighwayNetwork : xt::Cloneable<HighwayNetwork> {
    private:

    public:
        HighwayNetwork(int num_classes /* classes */, int in_channels = 3/* input channels */);

        HighwayNetwork(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


}