#include "../../../include/models/cnn/vggnet16.h"

namespace torch::ext::models {
    VggNet16::VggNet16(int num_classes) {
        //TODO layer1 DONE
        layer1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );


        //TODO layer2 DONE
        layer2 = torch::nn::Sequential();
        //        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).stride(1).padding(1));
        layer2->push_back(conv2);
        //        nn.BatchNorm2d(64),
        torch::nn::BatchNorm2d batch2 = torch::nn::BatchNorm2d(64);
        layer2->push_back(batch2);
        //        nn.ReLU(),
        torch::nn::ReLU relu2 = torch::nn::ReLU();
        layer2->push_back(relu2);
        //        nn.MaxPool2d(kernel_size = 2, stride = 2))
        torch::nn::MaxPool2d pool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer2->push_back(pool2);

        //TODO layer3 DONE
        layer3 = torch::nn::Sequential();
        //        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(1).padding(1));
        layer3->push_back(conv3);
        //        nn.BatchNorm2d(128),
        torch::nn::BatchNorm2d batch3 = torch::nn::BatchNorm2d(128);
        layer3->push_back(batch3);
        //        nn.ReLU())
        torch::nn::ReLU relu3 = torch::nn::ReLU();
        layer3->push_back(relu3);

        //TODO layer4 DONE
        layer4 = torch::nn::Sequential();
        //        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).stride(1).padding(1));
        layer4->push_back(conv4);
        //        nn.BatchNorm2d(128),
        torch::nn::BatchNorm2d batch4 = torch::nn::BatchNorm2d(128);
        layer4->push_back(batch4);
        //        nn.ReLU(),
        torch::nn::ReLU relu4 = torch::nn::ReLU();
        layer4->push_back(relu4);
        //        nn.MaxPool2d(kernel_size = 2, stride = 2))
        torch::nn::MaxPool2d pool4 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer4->push_back(pool4);


        //TODO layer5 DONE
        layer5 = torch::nn::Sequential();
        //        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).stride(1).padding(1));
        layer5->push_back(conv5);
        //        nn.BatchNorm2d(256),
        torch::nn::BatchNorm2d batch5 = torch::nn::BatchNorm2d(256);
        layer5->push_back(batch5);
        //        nn.ReLU())
        torch::nn::ReLU relu5 = torch::nn::ReLU();
        layer5->push_back(relu5);


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

    torch::Tensor VggNet16::forward(torch::Tensor x) {
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
