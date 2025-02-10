#include "../../../include/models/cnn/resnet.h"


ResidualBlock::ResidualBlock(int in_channels, int out_channels, int stride, torch::nn::Sequential downsample) {
    conv1 = torch::nn::Sequential();
    //                nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
    torch::nn::Conv2d cnv1 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1));
    conv1->push_back(cnv1);
    //                nn.BatchNorm2d(out_channels),
    torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(out_channels);
    conv1->push_back(batch1);
    //                nn.ReLU())
    torch::nn::ReLU relu1 = torch::nn::ReLU();
    conv1->push_back(relu1);
    register_module("conv1", conv1);

    conv2 = torch::nn::Sequential();
    //                nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
    torch::nn::Conv2d cnv2 = torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1));
    conv2->push_back(cnv2);

    //                nn.BatchNorm2d(out_channels),
    torch::nn::BatchNorm2d batch2 = torch::nn::BatchNorm2d(out_channels);
    conv2->push_back(batch2);
    this->downsample = downsample;
    this->relu = torch::nn::ReLU();
    this->out_channels = out_channels;
}

torch::Tensor ResidualBlock::forward(torch::Tensor x) {
    residual = x;
    torch::Tensor out = conv1->forward(x);
    out = conv2->forward(out);
    if (downsample) {
        residual = downsample->forward(x);
    } else {}
    out += residual;
    out = relu(out);
    return out;
}


ResNet::ResNet(vector<int> layers, int num_classes) {
    inplanes = 64;
    conv1 = torch::nn::Sequential();
//                nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
    torch::nn::Conv2d cnv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3));
    conv1->push_back(cnv1);
//                nn.BatchNorm2d(64),
    torch::nn::BatchNorm2d batch1 = torch::nn::BatchNorm2d(64);
    conv1->push_back(batch1);
    //                nn.ReLU())
    torch::nn::ReLU relu1 = torch::nn::ReLU();
    conv1->push_back(relu1);
    register_module("conv1", conv1);

    maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
    maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
    layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
    register_module("layer0", layer0);
    layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
    register_module("layer1", layer1);
    layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
    register_module("layer2", layer2);
    layer3 = makeLayerFromResidualBlock(512, layers[3], 2);
    register_module("layer3", layer3);
    avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
    fc = torch::nn::Linear(512, num_classes);
}

torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    torch::nn::Sequential downsample = nullptr;
    if (stride != 1 || inplanes != planes) {
        downsample = torch::nn::Sequential();
        //                    nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
        torch::nn::Conv2d convd = torch::nn::Conv2d(
                torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride).padding(0));
        downsample->push_back(convd);
        //                    nn.BatchNorm2d(planes),
        torch::nn::BatchNorm2d batchd = torch::nn::BatchNorm2d(planes);
        downsample->push_back(batchd);
    }
    torch::nn::Sequential layers = torch::nn::Sequential();
    ResidualBlock rb = ResidualBlock(inplanes, planes, stride, downsample);
    layers->push_back(rb);
    inplanes = planes;
    for (int i = 1; i < blocks; i++) {
        ResidualBlock rbt = ResidualBlock(inplanes, planes);
        layers->push_back(rbt);
    }
    return layers;
}

torch::Tensor ResNet::forward(torch::Tensor x) {
    x = conv1->forward(x);
    x = maxpool->forward(x);
    x = layer0->forward(x);
    x = layer1->forward(x);
    x = layer2->forward(x);
    x = layer3->forward(x);
    x = avgpool->forward(x);
    x = x.view({x.size(0), -1});
    x = fc->forward(x);
    return x;
}


