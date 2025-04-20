#include "../../../../include/models/cnn/unet/unet.h"


namespace xt::models {
    DoubleConv::DoubleConv(int in_channels, int out_channels) {
        this->conv_op = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).padding(1)),
            torch::nn::ReLU(),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).padding(1)),
            torch::nn::ReLU()
        );


        register_module("conv_op", this->conv_op);
    }

    torch::Tensor DoubleConv::forward(torch::Tensor input) {
        return this->conv_op->forward(input);
    }


    DownSample::DownSample(int in_channels, int out_channels) {
        this->conv = DoubleConv(in_channels, out_channels);
        this->pool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2));
        // register_module("conv", conv);
        register_module("conv", std::make_shared<DoubleConv>(conv));
        register_module("pool", pool);
    }

    torch::Tensor DownSample::forward(torch::Tensor input) {
        torch::Tensor x = this->conv.forward(input);
        x = this->pool->forward(x);
        return x;
    }

    //        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
    //        self.conv = DoubleConv(in_channels, out_channels)

    UpSample::UpSample(int in_channels, int out_channels) {
        this->conv = DoubleConv(in_channels, out_channels);
        this->up = torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(in_channels, in_channels / 2, 2).stride(2));

        register_module("conv", std::make_shared<DoubleConv>(conv));
        register_module("up", up);
    }

    torch::Tensor UpSample::forward(torch::Tensor x1, torch::Tensor x2) {
        x1 = this->up(x1);
        torch::Tensor x = torch::cat({x1, x2});
        return this->conv.forward(x);
        // def forward(self, x1, x2):
        //     x1 = self.up(x1)
        //     x = torch.cat([x1, x2], 1)
        //     return self.conv(x)
    }


    UNet::UNet(int num_classes, int in_channels) {

        this->down_convolution_1 = DownSample(in_channels, 64);
        this->down_convolution_2 = DownSample(64, 128);
        this->down_convolution_3 = DownSample(128, 256);
        this->down_convolution_4 = DownSample(256, 512);

        this->bottle_neck = DoubleConv(512, 1024);

        this->up_convolution_1 = UpSample(1024, 512);
        this->up_convolution_2 = UpSample(512, 256);
        this->up_convolution_3 = UpSample(256, 128);
        this->up_convolution_4 = UpSample(128, 64);

        this->out = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, num_classes, 1));
        //
        //        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    }


    torch::Tensor UNet::forward(torch::Tensor input) const {
        return input;
    }
}
