#include "../../../include/models/cnn/mobilenet.h"

namespace xt::models {
    HSigmoid::HSigmoid() {
        this->relu6 = torch::nn::ReLU6(torch::nn::ReLU6Options(true));
    }

    torch::Tensor HSigmoid::forward(torch::Tensor x) {
        x = this->relu6(x + 3) / 6;
        return x;
    }

    // --------------------------------------------------------------------

    HSwish::HSwish() {
        this->relu6 = torch::nn::ReLU6(torch::nn::ReLU6Options(true));
    }

    torch::Tensor HSwish::forward(torch::Tensor x) {
        x = this->relu6(x + 3) / 6;
        return x;
    }

    // --------------------------------------------------------------------

    SqueezeExcite::SqueezeExcite(int input_channels, int squeeze) {
        this->SE = torch::nn::Sequential(
            torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(1)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_channels, input_channels / squeeze, 1).stride(1).bias(false)),
            torch::nn::BatchNorm2d(input_channels / squeeze),
            torch::nn::ReLU(torch::nn::ReLUOptions(true)),
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_channels / squeeze, input_channels, 1).stride(1).bias(false)),
            torch::nn::BatchNorm2d(input_channels),
            HSigmoid::HSigmoid()
        );
    }

    torch::Tensor SqueezeExcite::forward(torch::Tensor x) {
        x = this->SE->forward(x);
        return x;
    }

    // --------------------------------------------------------------------

    Bottleneck::Bottleneck(int input_channels, int kernel, int stride, int expansion, int output_channels,
                           torch::nn::Module activation, bool se) {


    }


    MobileNetV3::MobileNetV3() {
        throw std::runtime_error("MobileNetV3::MobileNetV3()");
    }
}
