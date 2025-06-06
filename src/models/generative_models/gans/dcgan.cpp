#include "include/models/generative_models/gans/dcgan.h"


using namespace std;

namespace xt::models
{
    DCGAN::DCGAN(int num_classes, int in_channels)
    {
    }

    DCGAN::DCGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DCGAN::reset()
    {
    }

    auto DCGAN::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }


    DCGAN::Generator::Generator(int nz, int ngf, int nc)
        : conv1(torch::nn::ConvTranspose2dOptions(nz, ngf * 8, 4).stride(1).padding(0).bias(false)),
          conv2(torch::nn::ConvTranspose2dOptions(ngf * 8, ngf * 4, 4).stride(2).padding(1).bias(false)),
          conv3(torch::nn::ConvTranspose2dOptions(ngf * 4, ngf * 2, 4).stride(2).padding(1).bias(false)),
          conv4(torch::nn::ConvTranspose2dOptions(ngf * 2, ngf, 4).stride(2).padding(1).bias(false)),
          conv5(torch::nn::ConvTranspose2dOptions(ngf, nc, 4).stride(2).padding(1).bias(false)),
          bn1(torch::nn::BatchNorm2d(ngf * 8)),
          bn2(torch::nn::BatchNorm2d(ngf * 4)),
          bn3(torch::nn::BatchNorm2d(ngf * 2)),
          bn4(torch::nn::BatchNorm2d(ngf))
    {
        register_module("conv1", conv1);
        register_module("conv2", conv2);
        register_module("conv3", conv3);
        register_module("conv4", conv4);
        register_module("conv5", conv5);
        register_module("bn1", bn1);
        register_module("bn2", bn2);
        register_module("bn3", bn3);
        register_module("bn4", bn4);

        // Initialize weights
        apply([](torch::nn::Module& module)
        {
            if (auto* conv = dynamic_cast<torch::nn::ConvTranspose2dImpl*>(&module))
            {
                torch::nn::init::normal_(conv->weight, 0.0, 0.02);
            }
            else if (auto* bn = dynamic_cast<torch::nn::BatchNorm2dImpl*>(&module))
            {
                torch::nn::init::normal_(bn->weight, 1.0, 0.02);
                torch::nn::init::constant_(bn->bias, 0.0);
            }
        });
    }

    torch::Tensor DCGAN::Generator::forward(torch::Tensor x)
    {
        x = torch::relu(bn1(conv1(x)));
        x = torch::relu(bn2(conv2(x)));
        x = torch::relu(bn3(conv3(x)));
        x = torch::relu(bn4(conv4(x)));
        x = torch::tanh(conv5(x));
        return x;
    }
}
