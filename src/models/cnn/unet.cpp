#include "../../../include/models/cnn/unet.h"


namespace xt::models {
    DoubleConv::DoubleConv(int in_channels, int out_channels) {
        this->conv_op = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)),
            torch::nn::ReLU()
        );
    }

    torch::Tensor DoubleConv::forward(torch::Tensor input) {
        return this->conv_op->forward(input);
    }


    UNet::UNet() {
        throw std::runtime_error("MobileNetV3::MobileNetV3()");
    }
}
