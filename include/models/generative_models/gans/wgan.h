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
            CriticImpl();

            torch::Tensor forward(torch::Tensor x);

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
