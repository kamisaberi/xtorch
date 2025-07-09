#pragma once

#include "../../common.h"


// #include "../../../exceptions/implementation.h"

namespace xt::models {
    // Double Convolution Block
    struct DoubleConvImpl : torch::nn::Module {
        DoubleConvImpl(int in_channels, int out_channels);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    };

    TORCH_MODULE(DoubleConv);

    // U-Net Model
    struct UNetImpl : torch::nn::Module {
        UNetImpl(int in_channels, int out_channels) {
            // Encoder
            enc1 = register_module("enc1", DoubleConv(in_channels, 64));
            enc2 = register_module("enc2", DoubleConv(64, 128));
            enc3 = register_module("enc3", DoubleConv(128, 256));

            // Bottleneck
            bottleneck = register_module("bottleneck", DoubleConv(256, 512));

            // Decoder
            upconv3 = register_module("upconv3", torch::nn::ConvTranspose2d(
                    torch::nn::ConvTranspose2dOptions(512, 256, 2).stride(2)));
            dec3 = register_module("dec3", DoubleConv(512, 256));
            upconv2 = register_module("upconv2", torch::nn::ConvTranspose2d(
                    torch::nn::ConvTranspose2dOptions(256, 128, 2).stride(2)));
            dec2 = register_module("dec2", DoubleConv(256, 128));
            upconv1 = register_module("upconv1", torch::nn::ConvTranspose2d(
                    torch::nn::ConvTranspose2dOptions(128, 64, 2).stride(2)));
            dec1 = register_module("dec1", DoubleConv(128, 64));

            // Output layer
            out_conv = register_module("out_conv", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(64, out_channels, 1)));
        }

        torch::Tensor forward(torch::Tensor x) {
            // x: [batch, in_channels, h, w]

            // Encoder
            auto e1 = enc1->forward(x); // [batch, 64, 28, 28]
            auto p1 = torch::max_pool2d(e1, 2); // [batch, 64, 14, 14]
            auto e2 = enc2->forward(p1); // [batch, 128, 14, 14]
            auto p2 = torch::max_pool2d(e2, 2); // [batch, 128, 7, 7]
            auto e3 = enc3->forward(p2); // [batch, 256, 7, 7]
            auto p3 = torch::max_pool2d(e3, 2); // [batch, 256, 3, 3]

            // Bottleneck
            auto b = bottleneck->forward(p3); // [batch, 512, 3, 3]

            // Decoder
            auto u3 = upconv3->forward(b); // [batch, 256, 6, 6]
            // Pad to match e3 size (7x7)
            u3 = torch::pad(u3, {0, 1, 0, 1}); // [batch, 256, 7, 7]
            auto d3 = dec3->forward(torch::cat({u3, e3}, 1)); // [batch, 256, 7, 7]
            auto u2 = upconv2->forward(d3); // [batch, 128, 14, 14]
            auto d2 = dec2->forward(torch::cat({u2, e2}, 1)); // [batch, 128, 14, 14]
            auto u1 = upconv1->forward(d2); // [batch, 64, 28, 28]
            auto d1 = dec1->forward(torch::cat({u1, e1}, 1)); // [batch, 64, 28, 28]

            // Output
            auto out = out_conv->forward(d1); // [batch, out_channels, 28, 28]
            return out;
        }

        DoubleConv enc1{nullptr}, enc2{nullptr}, enc3{nullptr}, bottleneck{nullptr};
        DoubleConv dec1{nullptr}, dec2{nullptr}, dec3{nullptr};
        torch::nn::ConvTranspose2d upconv1{nullptr}, upconv2{nullptr}, upconv3{nullptr};
        torch::nn::Conv2d out_conv{nullptr};
    };

    TORCH_MODULE(UNet);

    // Dice Loss for Binary Segmentation
    struct DiceLossImpl : torch::nn::Module {
        DiceLossImpl(float smooth = 1.0) : smooth_(smooth) {
        }

        torch::Tensor forward(torch::Tensor input, torch::Tensor target) {
            // input: [batch, 1, h, w], target: [batch, 1, h, w]
            input = torch::sigmoid(input); // Convert logits to probabilities
            auto intersection = (input * target).sum({2, 3}); // [batch, 1]
            auto union1 = input.sum({2, 3}) + target.sum({2, 3}); // [batch, 1]
            auto dice = (2.0 * intersection + smooth_) / (union1 + smooth_); // [batch, 1]
            return 1.0 - dice.mean(); // Average loss
        }

        float smooth_;
    };

    TORCH_MODULE(DiceLoss);
}
