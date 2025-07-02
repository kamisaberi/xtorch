#include "include/models/computer_vision/image_classification/vggnet.h"

namespace xt::models
{
    VggNet11::VggNet11(int num_classes, int in_channels)
    {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        fc = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(7 * 7 * 512, 4096),
            torch::nn::ReLU()
        );

        fc1 = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU()
        );

        fc2 = torch::nn::Sequential(torch::nn::Linear(4096, num_classes));

        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("layer6", layer6);
        register_module("layer7", layer7);
        register_module("layer8", layer8);
        register_module("layer9", layer9);
        register_module("layer10", layer10);
        register_module("layer11", layer11);
        register_module("layer12", layer12);
        register_module("layer13", layer13);
        register_module("fc", fc);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        reset();
    }


    VggNet11::VggNet11(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        // Register conv layers
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("layer6", layer6);
        register_module("layer7", layer7);
        register_module("layer8", layer8);
        register_module("layer9", layer9);
        register_module("layer10", layer10);
        register_module("layer11", layer11);
        register_module("layer12", layer12);
        register_module("layer13", layer13);

        // Dynamically compute flattened output size
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = dummy_input;
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        int flattened_size = x.numel() / x.size(0);

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


    auto VggNet11::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        x = fc1->forward(x);
        x = fc2->forward(x);
        return x;
    }
    void VggNet11::reset()
    {
    }



    VggNet13::VggNet13(int num_classes, int in_channels)
    {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        fc = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(7 * 7 * 512, 4096),
            torch::nn::ReLU()
        );

        fc1 = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU()
        );

        fc2 = torch::nn::Sequential(torch::nn::Linear(4096, num_classes));

        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("layer6", layer6);
        register_module("layer7", layer7);
        register_module("layer8", layer8);
        register_module("layer9", layer9);
        register_module("layer10", layer10);
        register_module("layer11", layer11);
        register_module("layer12", layer12);
        register_module("layer13", layer13);
        register_module("fc", fc);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        reset();
    }


    VggNet13::VggNet13(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );

        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        // Register conv layers
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("layer6", layer6);
        register_module("layer7", layer7);
        register_module("layer8", layer8);
        register_module("layer9", layer9);
        register_module("layer10", layer10);
        register_module("layer11", layer11);
        register_module("layer12", layer12);
        register_module("layer13", layer13);

        // Dynamically compute flattened output size
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = dummy_input;
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        int flattened_size = x.numel() / x.size(0);

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



    auto VggNet13::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        x = fc1->forward(x);
        x = fc2->forward(x);
        return x;
    }
    void VggNet13::reset()
    {
    }



    VggNet16::VggNet16(int num_classes, int in_channels)
    {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        fc = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(7 * 7 * 512, 4096),
            torch::nn::ReLU()
        );

        fc1 = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU()
        );

        fc2 = torch::nn::Sequential(torch::nn::Linear(4096, num_classes));

        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("layer6", layer6);
        register_module("layer7", layer7);
        register_module("layer8", layer8);
        register_module("layer9", layer9);
        register_module("layer10", layer10);
        register_module("layer11", layer11);
        register_module("layer12", layer12);
        register_module("layer13", layer13);
        register_module("fc", fc);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        reset();
    }


    VggNet16::VggNet16(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        // Register conv layers
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("layer6", layer6);
        register_module("layer7", layer7);
        register_module("layer8", layer8);
        register_module("layer9", layer9);
        register_module("layer10", layer10);
        register_module("layer11", layer11);
        register_module("layer12", layer12);
        register_module("layer13", layer13);

        // Dynamically compute flattened output size
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = dummy_input;
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        int flattened_size = x.numel() / x.size(0);

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


    auto VggNet16::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        x = fc1->forward(x);
        x = fc2->forward(x);
        return x;
    }
    void VggNet16::reset()
    {
    }



    VggNet19::VggNet19(int num_classes, int in_channels)
    {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        fc = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(7 * 7 * 512, 4096),
            torch::nn::ReLU()
        );

        fc1 = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU()
        );

        fc2 = torch::nn::Sequential(torch::nn::Linear(4096, num_classes));

        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("layer6", layer6);
        register_module("layer7", layer7);
        register_module("layer8", layer8);
        register_module("layer9", layer9);
        register_module("layer10", layer10);
        register_module("layer11", layer11);
        register_module("layer12", layer12);
        register_module("layer13", layer13);
        register_module("fc", fc);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
        reset();
    }


    VggNet19::VggNet19(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        // Register conv layers
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);
        register_module("layer4", layer4);
        register_module("layer5", layer5);
        register_module("layer6", layer6);
        register_module("layer7", layer7);
        register_module("layer8", layer8);
        register_module("layer9", layer9);
        register_module("layer10", layer10);
        register_module("layer11", layer11);
        register_module("layer12", layer12);
        register_module("layer13", layer13);

        // Dynamically compute flattened output size
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = dummy_input;
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        int flattened_size = x.numel() / x.size(0);

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



    auto VggNet19::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        x = layer5->forward(x);
        x = layer6->forward(x);
        x = layer7->forward(x);
        x = layer8->forward(x);
        x = layer9->forward(x);
        x = layer10->forward(x);
        x = layer11->forward(x);
        x = layer12->forward(x);
        x = layer13->forward(x);
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        x = fc1->forward(x);
        x = fc2->forward(x);
        return x;
    }

    void VggNet19::reset()
    {
    }
}
