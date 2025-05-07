#include "../../../../include/models/computer_vision/image_classification/vggnet19.h"

namespace xt::models {
    VggNet19::VggNet19(int num_classes,int in_channels) : BaseModel() {
        //TODO layer1 DONE
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        //TODO layer2 DONE
        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        //TODO layer3 DONE
        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        //TODO layer4 DONE
        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        //TODO layer5 DONE
        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        //TODO layer6 DONE
        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        //TODO layer7 DONE
        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        //TODO layer8 DONE
        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        //TODO layer9 DONE
        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        //TODO layer10 DONE
        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        //TODO layer11 DONE
        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        //TODO layer12 DONE
        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        //TODO layer13
        layer13 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        //TODO fc DONE
        fc = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(7 * 7 * 512, 4096),
            torch::nn::ReLU()
        );

        //TODO fc1 DONE
        fc1 = torch::nn::Sequential(
            torch::nn::Dropout(0.5),
            torch::nn::Linear(4096, 4096),
            torch::nn::ReLU()
        );

        //TODO fc2 DONE
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
    }


    VggNet19::VggNet19(int num_classes, int in_channels, std::vector<int64_t> input_shape)
        : BaseModel() {

        //TODO layer1 DONE
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        //TODO layer2 DONE
        layer2 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );

        //TODO layer3 DONE
        layer3 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU()
        );

        //TODO layer4 DONE
        layer4 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        //TODO layer5 DONE
        layer5 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()
        );


        //TODO layer6 DONE
        layer6 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU()

        );


        //TODO layer7 DONE
        layer7 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(256),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        //TODO layer8 DONE
        layer8 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        //TODO layer9 DONE
        layer9 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        //TODO layer10 DONE
        layer10 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2))
        );


        //TODO layer11 DONE
        layer11 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        //TODO layer12 DONE
        layer12 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(512),
            torch::nn::ReLU()
        );


        //TODO layer13
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
    }




    torch::Tensor VggNet19::forward(torch::Tensor x) const {
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
}
