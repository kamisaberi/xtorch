#include "../../../include/models/cnn/lenet5.h"


using namespace std;
namespace torch::ext::models {


    LeNet5::LeNet5(int num_classes) {
        layer1 = torch::nn::Sequential();
        torch::nn::Conv2d conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, 5).stride(1).padding(0));
        layer1->push_back(conv1);
        torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(6);
        layer1->push_back(batch1);
        torch::nn::ReLU relu1 = torch::nn::ReLU();
        layer1->push_back(relu1);
        torch::nn::MaxPool2d pool1 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer1->push_back(pool1);
        register_module("layer1", layer1);

        layer2 = torch::nn::Sequential();
        torch::nn::Conv2d conv2 = torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, 5).stride(1).padding(0));
        layer2->push_back(conv2);
        torch::nn::BatchNorm2d batch2 = torch::nn::BatchNorm2d(16);
        layer2->push_back(batch2);
        torch::nn::ReLU relu2 = torch::nn::ReLU();
        layer2->push_back(relu2);
        torch::nn::MaxPool2d pool2 = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        layer2->push_back(pool2);
        register_module("layer2", layer2);

        fc1 = torch::nn::Linear(400, 120);
        fc2 = torch::nn::Linear(120, 84);
        fc3 = torch::nn::Linear(84, num_classes);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
        register_module("fc3", fc3);
    }

    torch::Tensor LeNet5::forward(torch::Tensor x) {
        cout << "le 01" << endl;
        x = layer1->forward(x);
        cout << "le 01-01" << endl;
        x = layer2->forward(x);
        cout << "le 02" << endl;
        x = torch::relu(fc1->forward(x.view({-1, 400})));
        x = torch::relu(fc2->forward(x));
        cout << "le 03" << endl;
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        cout << "le 04" << endl;
        return x;
    }


}