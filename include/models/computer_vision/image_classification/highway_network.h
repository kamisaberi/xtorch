#pragma once

#include "../../common.h"

namespace xt::models
{
    // Highway Layer
    struct HighwayLayer : xt::Module
    {
        HighwayLayer(int input_size);
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Linear transform{nullptr}, plain{nullptr};
    };

    // TORCH_MODULE(HighwayLayer);

    // Highway Network
    struct HighwayNetwork : xt::Module
    {
        HighwayNetwork(int input_size, int num_classes, int num_layers);
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Linear input_layer{nullptr}, output_layer{nullptr};
        vector< std::shared_ptr<xt::Module>> layers;
    };

    // TORCH_MODULE(HighwayNetwork);


    // struct HighwayNetwork : xt::Cloneable<HighwayNetwork> {
    // private:
    //
    // public:
    //     HighwayNetwork(int num_classes /* classes */, int in_channels = 3/* input channels */);
    //
    //     HighwayNetwork(int num_classes, int in_channels, std::vector <int64_t> input_shape);
    //
    //     auto forward(std::initializer_list <std::any> tensors) -> std::any override;
    //
    //     void reset() override;
    // };
}
