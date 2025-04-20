#include "../../../../include/models/cnn/lenet/lenet5.h"


using namespace std;

namespace xt::models {


    LeNet5::LeNet5(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    : BaseModel() {

        // Define convolutional layers
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 6, 5).stride(1).padding(0)),
            torch::nn::BatchNorm2d(6),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5).stride(1).padding(0)),
            torch::nn::BatchNorm2d(16),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        // Calculate the output size after conv layers dynamically
        torch::NoGradGuard no_grad;

        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto out = layer1->forward(dummy_input);
        out = layer2->forward(out);

        int flattened_size = out.numel() / out.size(0);  // total features per sample

        // Define fully connected layers
        fc1 = torch::nn::Linear(flattened_size, 120);
        fc2 = torch::nn::Linear(120, 84);
        fc3 = torch::nn::Linear(84, num_classes);

        // Register modules
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }


    LeNet5::LeNet5(int num_classes, int in_channels) : BaseModel() {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 6, 5).stride(1).padding(0)),
            torch::nn::BatchNorm2d(6),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5).stride(1).padding(0)),
            torch::nn::BatchNorm2d(16),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        fc1 = torch::nn::Linear(400, 120);
        fc2 = torch::nn::Linear(120, 84);
        fc3 = torch::nn::Linear(84, num_classes);


        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    // torch::Tensor LeNet5::forward(torch::Tensor x) const {
    //     // cout << "t1\n";
    //     x = layer1->forward(x);
    //     x = layer2->forward(x);
    //     x = torch::relu(fc1->forward(x.view({-1, 400})));
    //     x = torch::relu(fc2->forward(x));
    //     x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    //     // cout << x.sizes() << endl;
    //     // cout << x << endl;
    //     return x;
    // }
    torch::Tensor LeNet5::forward(torch::Tensor x) const {
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = x.view({x.size(0), -1}); // flatten
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), 1);
        return x;
    }

}

/*
    LeNet5::LeNet5(int num_classes, int in_channels, std::vector<int64_t> input_shape)
        : BaseModel() {

        // Define convolutional layers
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 6, 5).stride(1).padding(0)),
            torch::nn::BatchNorm2d(6),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5).stride(1).padding(0)),
            torch::nn::BatchNorm2d(16),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        // Calculate the output size after conv layers dynamically
        torch::NoGradGuard no_grad;

        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto out = layer1->forward(dummy_input);
        out = layer2->forward(out);

        int flattened_size = out.numel() / out.size(0);  // total features per sample

        // Define fully connected layers
        fc1 = torch::nn::Linear(flattened_size, 120);
        fc2 = torch::nn::Linear(120, 84);
        fc3 = torch::nn::Linear(84, num_classes);

        // Register modules
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor LeNet5::forward(torch::Tensor x) const {
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = x.view({x.size(0), -1}); // flatten
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), 1);
        return x;
    }
 */