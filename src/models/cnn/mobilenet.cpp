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

    torch::Tensor Bottleneck::forward(torch::Tensor x) {
        return x;
    }


    // --------------------------------------------------------------------

    MobileNetV3::MobileNetV3(int input_channels, int num_classes, float dropout_prob) {
        this->initial_conv = torch::nn::Sequential(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(input_channels, 16, 3).stride(2)),
            torch::nn::BatchNorm2d(16),
            HSwish()
        );

        this->bottlenecks = torch::nn::Sequential(
            Bottleneck(16, 3, 1, 16, 16, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
            Bottleneck(16, 3, 2, 64, 24, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
            Bottleneck(24, 3, 1, 72, 24, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
            Bottleneck(24, 5, 2, 72, 40, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
            Bottleneck(40, 5, 1, 120, 40, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
            Bottleneck(40, 5, 1, 120, 40, static_cast<Module>(*torch::nn::ReLU(torch::nn::ReLUOptions(true)))),
            Bottleneck(40, 3, 2, 240, 80, static_cast<Module>(HSwish())),
            Bottleneck(80, 3, 1, 200, 80, static_cast<Module>(HSwish())),
            Bottleneck(80, 3, 1, 184, 80, static_cast<Module>(HSwish())),
            Bottleneck(80, 3, 1, 184, 80, static_cast<Module>(HSwish())),
            Bottleneck(80, 3, 1, 480, 112, static_cast<Module>(HSwish()), true),
            Bottleneck(112, 3, 1, 672, 112, static_cast<Module>(HSwish()), true),
            Bottleneck(112, 5, 2, 672, 160, static_cast<Module>(HSwish()), true),
            Bottleneck(160, 5, 1, 960, 160, static_cast<Module>(HSwish()), true),
            Bottleneck(160, 5, 1, 960, 160, static_cast<Module>(HSwish()), true)
        );


        this->final_conv = torch::nn::Sequential(
            torch::nn::Conv2d(
                torch::nn::Conv2dOptions(160, 960, 1).stride(1).bias(false)),
            torch::nn::BatchNorm2d(960),
            HSwish()
        );


        this->pool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions(2));


        this->classifier = torch::nn::Sequential(
            torch::nn::Linear(960, 1200),
            HSwish(),
            torch::nn::Dropout(torch::nn::DropoutOptions(dropout_prob).inplace(true)),
            torch::nn::Linear(1200, num_classes)
        );
    }

    torch::Tensor MobileNetV3::forward(torch::Tensor x) {
        x = this->initial_conv->forward(x);
        x = this->bottlenecks->forward(x);
        x = this->final_conv->forward(x);
        x = this->pool(x);
        x = x.view({x.size(0), -1});
        //        x = torch.flatten(x, 1)
        x = this->classifier->forward(x);
        return x;
    }
}
