#pragma once
#include "../../common.h"


namespace xt::models
{
    struct WGANGP : xt::Cloneable<WGANGP>
    {
    private:

    public:
        // Generator Network
        struct GeneratorImpl : torch::nn::Module
        {
            GeneratorImpl(int latent_dim);

            torch::Tensor forward(torch::Tensor z);

            torch::nn::Linear fc{nullptr};
            torch::nn::ConvTranspose2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
            torch::nn::ReLU relu{nullptr};
        };

        TORCH_MODULE(Generator);

        // Critic Network
        struct CriticImpl : torch::nn::Module
        {
            CriticImpl();

            torch::Tensor forward(torch::Tensor x);

            torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
            torch::nn::Linear fc{nullptr};
            torch::nn::LeakyReLU lrelu{nullptr};
        };

        TORCH_MODULE(Critic);


        WGANGP(int num_classes /* classes */, int in_channels = 3/* input channels */);

        WGANGP(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
