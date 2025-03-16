#include "../../../include/models/cnn/alexnet.h"


using namespace std;

namespace torch::ext::models {
    AlexNet::AlexNet(int num_classes , int in_channels) {

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
    }

    torch::Tensor AlexNet::forward(torch::Tensor x) {
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
}
