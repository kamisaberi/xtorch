#include "../../../include/models/cnn/alexnet.h"


using namespace std;

namespace torch::ext::cnn::models {

    AlexNet::AlexNet(int num_classes) {
        //TODO layer1
        layer1 = torch::nn::Sequential();
        //             nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
        torch::nn::Conv2d conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 96, 11).stride(4).padding(0));
        layer1->push_back(conv1);
        //             nn.BatchNorm2d(96),
        torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(96);
        layer1->push_back(batch1);
        //             nn.ReLU(),
        torch::nn::ReLU relu1 = torch::nn::ReLU();
        layer1->push_back(relu1);
        //             nn.MaxPool2d(kernel_size = 3, stride = 2))
        torch::nn::MaxPool2d pool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        layer1->push_back(pool1);
        register_module("layer1", layer1);

        //TODO layer2
        layer2 = torch::nn::Sequential();
        //             nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
        torch::nn::Conv2d conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(96, 256, 5).stride(1).padding(2));
        layer2->push_back(conv2);
        //             nn.BatchNorm2d(256),
        torch::nn::BatchNorm2d batch2 = torch::nn::BatchNorm2d(256);
        layer2->push_back(batch2);
        //             nn.ReLU(),
        torch::nn::ReLU relu2 = torch::nn::ReLU();
        layer2->push_back(relu2);
        //             nn.MaxPool2d(kernel_size = 3, stride = 2))
        torch::nn::MaxPool2d pool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        layer2->push_back(pool2);
        register_module("layer2", layer2);

        //TODO layer3
        layer3 = torch::nn::Sequential();
        //             nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv3 = torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 384, 3).stride(1).padding(1));
        layer3->push_back(conv3);
        //             nn.BatchNorm2d(384),
        torch::nn::BatchNorm2d batch3 = torch::nn::BatchNorm2d(384);
        layer3->push_back(batch3);
        //             nn.ReLU())
        torch::nn::ReLU relu3 = torch::nn::ReLU();
        layer3->push_back(relu3);
        register_module("layer3", layer3);

        //TODO layer4
        layer4 = torch::nn::Sequential();
        //             nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv4 = torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 384, 3).stride(1).padding(1));
        layer4->push_back(conv4);
        //             nn.BatchNorm2d(384),
        torch::nn::BatchNorm2d batch4 = torch::nn::BatchNorm2d(384);
        layer4->push_back(batch4);
        //             nn.ReLU())
        torch::nn::ReLU relu4 = torch::nn::ReLU();
        layer4->push_back(relu4);
        register_module("layer4", layer4);

        //TODO layer5
        layer5 = torch::nn::Sequential();
        //             nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
        torch::nn::Conv2d conv5 = torch::nn::Conv2d(torch::nn::Conv2dOptions(384, 256, 3).stride(1).padding(1));
        layer5->push_back(conv5);
        //             nn.BatchNorm2d(256),
        torch::nn::BatchNorm2d batch5 = torch::nn::BatchNorm2d(256);
        layer5->push_back(batch5);
        //             nn.ReLU(),
        torch::nn::ReLU relu5 = torch::nn::ReLU();
        layer5->push_back(relu5);
        //             nn.MaxPool2d(kernel_size = 3, stride = 2))
        torch::nn::MaxPool2d pool5 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2));
        layer5->push_back(pool5);
        register_module("layer5", layer5);

        //TODO fc
        fc = torch::nn::Sequential();
        //             nn.Dropout(0.5),
        torch::nn::Dropout drop10 = torch::nn::Dropout(0.5);
        fc->push_back(drop10);
        //             nn.Linear(9216, 4096),
        torch::nn::Linear linear10 = torch::nn::Linear(9216, 4096);
        fc->push_back(linear10);
        //             nn.ReLU())
        torch::nn::ReLU relu10 = torch::nn::ReLU();
        fc->push_back(relu10);
        register_module("fc", fc);


        //TODO fc1
        fc1 = torch::nn::Sequential();
        //             nn.Dropout(0.5),
        torch::nn::Dropout drop11 = torch::nn::Dropout(0.5);
        fc1->push_back(drop11);
        //             nn.Linear(4096, 4096),
        torch::nn::Linear linear11 = torch::nn::Linear(4096, 4096);
        fc1->push_back(linear11);
        //             nn.ReLU())
        torch::nn::ReLU relu11 = torch::nn::ReLU();
        fc1->push_back(relu11);
        register_module("fc1", fc1);

        //TODO fc2
        fc2 = torch::nn::Sequential();
        //             nn.Linear(4096, num_classes))
        torch::nn::Linear linear12 = torch::nn::Linear(4096, num_classes);
        fc2->push_back(linear12);
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