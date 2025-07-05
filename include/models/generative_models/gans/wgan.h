#pragma once
#include "../../common.h"


namespace xt::models
{
    struct WGAN : xt::Cloneable<WGAN>
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

        // Weight Clipping for Lipschitz Constraint
        void clip_weights(Critic& critic, float clip_value)
        {
            for (auto& param : critic->parameters())
            {
                param.data().clamp_(-clip_value, clip_value);
            }
        }


        WGAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        WGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
        void reset() override;
    };
}
