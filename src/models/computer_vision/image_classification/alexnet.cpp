#include "models/computer_vision/image_classification/alexnet.h"


using namespace std;

namespace xt::models
{
    AlexNet::AlexNet(int num_classes, int in_channels)
    {
        //TODO layer1
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 96, 11).stride(4).padding(0)),
            torch::nn::BatchNorm2d(96),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))
        );

        //TODO layer2
        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 256, 5).stride(1).padding(2)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))
        );


        //TODO layer3
        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 384, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(384),
            torch::nn::ReLU()
        );

        //TODO layer4
        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 384, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(384),
            torch::nn::ReLU()
        );

        //TODO layer5
        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))
        );


        //TODO fc
        fc = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(9216, 4096),
            torch::nn::ReLU()
        );


        //TODO fc1
        fc1 = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU()
        );

        //TODO fc2
        fc2 = torch::nn::Sequential(torch::nn::Linear(4096, num_classes));


        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("fc", fc);
        register_module("fc1", fc1);
        register_module("fc2", fc2);

        //TODO DONE
        reset();
    }

    AlexNet::AlexNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
        // Layer definitions
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 96, 11).stride(4).padding(0)),
            torch::nn::BatchNorm2d(96),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))
        );

        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 256, 5).stride(1).padding(2)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 384, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(384),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 384, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(384),
            torch::nn::ReLU()
        );

        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2))
        );

        // Register convolutional layers
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);

        // Dynamically compute the flattened feature size
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto out = layer1->forward(dummy_input);
        out = layer2->forward(out);
        out = layer3->forward(out);
        out = layer4->forward(out);
        out = layer5->forward(out);
        int64_t flattened_size = out.numel() / out.size(0);

        // Fully connected layers
        fc = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(flattened_size, 4096),
            torch::nn::ReLU()
        );

        fc1 = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU()
        );

        fc2 = torch::nn::Sequential(
            torch::nn::Linear(4096, num_classes)
        );

        register_module("fc", fc);
        register_module("fc1", fc1);
        register_module("fc2", fc2);

        reset();
    }


    torch::Tensor AlexNet::forward(torch::Tensor x) const
    {
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        x = fc1->forward(x);
        x = fc2->forward(x);
        return x;
    }

    void AlexNet::reset()
    {
    }

}
