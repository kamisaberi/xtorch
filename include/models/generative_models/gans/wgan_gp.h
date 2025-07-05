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
            GeneratorImpl(int latent_dim)
            {
                fc = register_module("fc", torch::nn::Linear(latent_dim, 256 * 4 * 4));
                conv1 = register_module("conv1", torch::nn::ConvTranspose2d(
                                            torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1)));
                conv2 = register_module("conv2", torch::nn::ConvTranspose2d(
                                            torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1)));
                conv3 = register_module("conv3", torch::nn::ConvTranspose2d(
                                            torch::nn::ConvTranspose2dOptions(64, 1, 3).stride(1).padding(1)));
                bn1 = register_module("bn1", torch::nn::BatchNorm2d(256));
                bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
                bn3 = register_module("bn3", torch::nn::BatchNorm2d(64));
                relu = register_module("relu", torch::nn::ReLU());
            }

            torch::Tensor forward(torch::Tensor z)
            {
                auto batch_size = z.size(0);
                z = relu->forward(fc->forward(z)); // [batch, 256 * 4 * 4]
                z = z.view({batch_size, 256, 4, 4}); // [batch, 256, 4, 4]
                z = relu->forward(bn1->forward(conv1->forward(z))); // [batch, 128, 8, 8]
                z = relu->forward(bn2->forward(conv2->forward(z))); // [batch, 64, 16, 16]
                z = torch::tanh(conv3->forward(bn3->forward(z))); // [batch, 1, 28, 28]
                return z;
            }

            torch::nn::Linear fc{nullptr};
            torch::nn::ConvTranspose2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
            torch::nn::ReLU relu{nullptr};
        };

        TORCH_MODULE(Generator);

        // Critic Network
        struct CriticImpl : torch::nn::Module
        {
            CriticImpl()
            {
                conv1 = register_module("conv1", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1)));
                conv2 = register_module("conv2", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
                conv3 = register_module("conv3", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1)));
                fc = register_module("fc", torch::nn::Linear(256 * 3 * 3, 1));
                lrelu = register_module("lrelu", torch::nn::LeakyReLU(
                                            torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            }

            torch::Tensor forward(torch::Tensor x)
            {
                x = lrelu->forward(conv1->forward(x)); // [batch, 64, 14, 14]
                x = lrelu->forward(conv2->forward(x)); // [batch, 128, 7, 7]
                x = lrelu->forward(conv3->forward(x)); // [batch, 256, 3, 3]
                auto batch_size = x.size(0);
                x = x.view({batch_size, -1}); // [batch, 256 * 3 * 3]
                x = fc->forward(x); // [batch, 1]
                return x;
            }

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
