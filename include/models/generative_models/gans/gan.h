#pragma once

#include "../../common.h"


namespace xt::models
{
    struct GAN : xt::Cloneable<GAN>
    {
    private:

    public:
        GAN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        GAN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;

        struct Generator : torch::nn::Module {
            Generator(int nz, int nhidden, int nout)
                : fc1(torch::nn::Linear(nz, nhidden)),
                  fc2(torch::nn::Linear(nhidden, nhidden)),
                  fc3(torch::nn::Linear(nhidden, nout)) {
                register_module("fc1", fc1);
                register_module("fc2", fc2);
                register_module("fc3", fc3);

                // Initialize weights
                apply([](torch::nn::Module& module) {
                    if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(&module)) {
                        torch::nn::init::normal_(linear->weight, 0.0, 0.02);
                        torch::nn::init::constant_(linear->bias, 0.0);
                    }
                });
            }

            torch::Tensor forward(torch::Tensor x) {
                x = torch::relu(fc1(x));
                x = torch::relu(fc2(x));
                x = torch::tanh(fc3(x));
                return x;
            }

            torch::nn::Linear fc1, fc2, fc3;
        };

        // Discriminator Network
        struct Discriminator : torch::nn::Module {
            Discriminator(int nin, int nhidden)
                : fc1(torch::nn::Linear(nin, nhidden)),
                  fc2(torch::nn::Linear(nhidden, nhidden)),
                  fc3(torch::nn::Linear(nhidden, 1)) {
                register_module("fc1", fc1);
                register_module("fc2", fc2);
                register_module("fc3", fc3);

                // Initialize weights
                apply([](torch::nn::Module& module) {
                    if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(&module)) {
                        torch::nn::init::normal_(linear->weight, 0.0, 0.02);
                        torch::nn::init::constant_(linear->bias, 0.0);
                    }
                });
            }

            torch::Tensor forward(torch::Tensor x) {
                x = torch::leaky_relu(fc1(x), 0.2);
                x = torch::leaky_relu(fc2(x), 0.2);
                x = torch::sigmoid(fc3(x));
                return x;
            }

            torch::nn::Linear fc1, fc2, fc3;
        };




    };

}