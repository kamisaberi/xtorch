#pragma once
#include "../../common.h"


namespace xt::models
{
    struct Pix2Pix : xt::Cloneable<Pix2Pix>
    {
    private:

    public:
        // U-Net Generator
        struct UNetGeneratorImpl : torch::nn::Module
        {
            UNetGeneratorImpl()
            {
                // Encoder
                enc1 = register_module("enc1", torch::nn::Conv2d(
                                           torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1)));
                // [batch, 64, 14, 14]
                enc2 = register_module("enc2", torch::nn::Conv2d(
                                           torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
                // [batch, 128, 7, 7]
                enc3 = register_module("enc3", torch::nn::Conv2d(
                                           torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1)));
                // [batch, 256, 3, 3]

                // Decoder
                dec3 = register_module("dec3", torch::nn::ConvTranspose2d(
                                           torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1)));
                // [batch, 128, 7, 7]
                dec2 = register_module("dec2", torch::nn::ConvTranspose2d(
                                           torch::nn::ConvTranspose2dOptions(256, 64, 4).stride(2).padding(1)));
                // [batch, 64, 14, 14]
                dec1 = register_module("dec1", torch::nn::ConvTranspose2d(
                                           torch::nn::ConvTranspose2dOptions(128, 1, 4).stride(2).padding(1)));
                // [batch, 1, 28, 28]

                bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
                bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
                bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
                bn4 = register_module("bn4", torch::nn::BatchNorm2d(128));
                bn5 = register_module("bn5", torch::nn::BatchNorm2d(64));
                relu = register_module("relu", torch::nn::ReLU());
                tanh = register_module("tanh", torch::nn::Tanh());
            }

            torch::Tensor forward(torch::Tensor x)
            {
                // Encoder
                auto e1 = relu->forward(bn1->forward(enc1->forward(x))); // [batch, 64, 14, 14]
                auto e2 = relu->forward(bn2->forward(enc2->forward(e1))); // [batch, 128, 7, 7]
                auto e3 = relu->forward(bn3->forward(enc3->forward(e2))); // [batch, 256, 3, 3]

                // Decoder with skip connections
                auto d3 = relu->forward(bn4->forward(dec3->forward(e3))); // [batch, 128, 7, 7]
                auto d2 = relu->forward(bn5->forward(dec2->forward(torch::cat({d3, e2}, 1)))); // [batch, 64, 14, 14]
                auto d1 = tanh->forward(dec1->forward(torch::cat({d2, e1}, 1))); // [batch, 1, 28, 28]

                return d1;
            }

            torch::nn::Conv2d enc1{nullptr}, enc2{nullptr}, enc3{nullptr};
            torch::nn::ConvTranspose2d dec3{nullptr}, dec2{nullptr}, dec1{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
            torch::nn::ReLU relu{nullptr};
            torch::nn::Tanh tanh{nullptr};
        };

        TORCH_MODULE(UNetGenerator);

        // PatchGAN Discriminator
        struct PatchGANDiscriminatorImpl : torch::nn::Module
        {
            PatchGANDiscriminatorImpl()
            {
                conv1 = register_module("conv1", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(2, 64, 4).stride(2).padding(1)));
                // [batch, 64, 14, 14]
                conv2 = register_module("conv2", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
                // [batch, 128, 7, 7]
                conv3 = register_module("conv3", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1)));
                // [batch, 256, 3, 3]
                conv4 = register_module("conv4", torch::nn::Conv2d(
                                            torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(1)));
                // [batch, 1, 3, 3]
                bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
                bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
                bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
                lrelu = register_module("lrelu", torch::nn::LeakyReLU(
                                            torch::nn::LeakyReLUOptions().negative_slope(0.2)));
            }

            torch::Tensor forward(torch::Tensor input, torch::Tensor target)
            {
                auto x = torch::cat({input, target}, 1); // [batch, 2, 28, 28]
                x = lrelu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
                x = lrelu->forward(bn2->forward(conv2->forward(x))); // [batch, 128, 7, 7]
                x = lrelu->forward(bn3->forward(conv3->forward(x))); // [batch, 256, 3, 3]
                x = torch::sigmoid(conv4->forward(x)); // [batch, 1, 3, 3]
                return x;
            }

            torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
            torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
            torch::nn::LeakyReLU lrelu{nullptr};
        };

        TORCH_MODULE(PatchGANDiscriminator);




        Pix2Pix(int num_classes /* classes */, int in_channels = 3/* input channels */);

        Pix2Pix(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
