#include "../../../include/models/cnn/mobilenet.h"

namespace xt::models {
    HSigmoid::HSigmoid() {
        this->relu6 = torch::nn::ReLU6();
    }

    torch::Tensor HSigmoid::forward(torch::Tensor x) {
        x = this->relu6(x + 3) / 6;
        return x;
    }

    // --------------------------------------------------------------------

    HSwish::HSwish() {
        this->relu6 = torch::nn::ReLU6();
    }

    torch::Tensor HSwish::forward(torch::Tensor x) {
        x = this->relu6(x + 3) / 6;
        return x;
    }

    // --------------------------------------------------------------------

    SqueezeExcite::SqueezeExcite(int input_channels, int squeeze) {


    }


    MobileNetV3::MobileNetV3() {
        throw std::runtime_error("MobileNetV3::MobileNetV3()");
    }
}
