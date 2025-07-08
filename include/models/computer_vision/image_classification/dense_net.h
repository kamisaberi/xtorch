#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models {

    // Dense Layer (Bottleneck: 1x1 conv -> 3x3 conv)
    struct DenseLayerImpl : torch::nn::Module {
        DenseLayerImpl(int in_channels, int growth_rate) {
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(in_channels));
            conv1 = register_module("conv1", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_channels, 4 * growth_rate, 1).bias(false)));
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(4 * growth_rate));
            conv2 = register_module("conv2", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(4 * growth_rate, growth_rate, 3).padding(1).bias(false)));
        }

        torch::Tensor forward(torch::Tensor x) {
            auto out = torch::relu(bn1->forward(x));
            out = conv1->forward(out);
            out = torch::relu(bn2->forward(out));
            out = conv2->forward(out);
            return torch::cat({x, out}, 1); // Concatenate input with output
        }

        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    };

    TORCH_MODULE(DenseLayer);

    // Dense Block
    struct DenseBlockImpl : torch::nn::Module {
        DenseBlockImpl(int num_layers, int in_channels, int growth_rate) {
            for (int i = 0; i < num_layers; ++i) {
                layers->push_back(DenseLayer(in_channels + i * growth_rate, growth_rate));
                register_module("denselayer_" + std::to_string(i), layers->back());
            }
        }

        torch::Tensor forward(torch::Tensor x) {
            for (auto &layer: *layers) {
                x = layer->forward(x);
            }
            return x;
        }

        torch::nn::ModuleList layers{torch::nn::ModuleList()};
    };

    TORCH_MODULE(DenseBlock);

    // Transition Layer (1x1 conv + 2x2 avg pool)
    struct TransitionLayerImpl : torch::nn::Module {
        TransitionLayerImpl(int in_channels, int out_channels) {
            bn = register_module("bn", torch::nn::BatchNorm2d(in_channels));
            conv = register_module("conv", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(in_channels, out_channels, 1).bias(false)));
            pool = register_module("pool", torch::nn::AvgPool2d(
                    torch::nn::AvgPool2dOptions(2).stride(2)));
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(bn->forward(x));
            x = conv->forward(x);
            x = pool->forward(x);
            return x;
        }

        torch::nn::BatchNorm2d bn{nullptr};
        torch::nn::Conv2d conv{nullptr};
        torch::nn::AvgPool2d pool{nullptr};
    };

    TORCH_MODULE(TransitionLayer);

    // DenseNet121
    struct DenseNet121Impl : torch::nn::Module {
        DenseNet121Impl(int num_classes = 10, int growth_rate = 32, int init_channels = 64) {
            // Initial conv layer
            conv0 = register_module("conv0", torch::nn::Conv2d(
                    torch::nn::Conv2dOptions(3, init_channels, 3).stride(1).padding(1).bias(false)));
            bn0 = register_module("bn0", torch::nn::BatchNorm2d(init_channels));

            // Dense blocks and transition layers
            int num_features = init_channels;
            dense1 = register_module("dense1", DenseBlock(/*num_layers*/6, num_features, growth_rate));
            num_features += 6 * growth_rate;
            trans1 = register_module("trans1", TransitionLayer(num_features, num_features / 2));
            num_features /= 2;

            dense2 = register_module("dense2", DenseBlock(/*num_layers*/12, num_features, growth_rate));
            num_features += 12 * growth_rate;
            trans2 = register_module("trans2", TransitionLayer(num_features, num_features / 2));
            num_features /= 2;

            dense3 = register_module("dense3", DenseBlock(/*num_layers*/24, num_features, growth_rate));
            num_features += 24 * growth_rate;
            trans3 = register_module("trans3", TransitionLayer(num_features, num_features / 2));
            num_features /= 2;

            dense4 = register_module("dense4", DenseBlock(/*num_layers*/16, num_features, growth_rate));
            num_features += 16 * growth_rate;

            // Final layers
            bn_final = register_module("bn_final", torch::nn::BatchNorm2d(num_features));
            fc = register_module("fc", torch::nn::Linear(num_features, num_classes));
        }

        torch::Tensor forward(torch::Tensor x) {
            x = torch::relu(bn0->forward(conv0->forward(x))); // [batch, 64, 32, 32]
            x = dense1->forward(x);
            x = trans1->forward(x); // [batch, num_features/2, 16, 16]
            x = dense2->forward(x);
            x = trans2->forward(x); // [batch, num_features/2, 8, 8]
            x = dense3->forward(x);
            x = trans3->forward(x); // [batch, num_features/2, 4, 4]
            x = dense4->forward(x);
            x = torch::relu(bn_final->forward(x));
            x = torch::avg_pool2d(x, x.size(2)); // Global avg pool: [batch, num_features, 1, 1]
            x = x.view({x.size(0), -1}); // [batch, num_features]
            x = fc->forward(x); // [batch, num_classes]
            return x;
        }

        torch::nn::Conv2d conv0{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn_final{nullptr};
        DenseBlock dense1{nullptr}, dense2{nullptr}, dense3{nullptr}, dense4{nullptr};
        TransitionLayer trans1{nullptr}, trans2{nullptr}, trans3{nullptr};
        torch::nn::Linear fc{nullptr};
    };

    TORCH_MODULE(DenseNet121);


    struct DenseNet121 : xt::Cloneable<DenseNet121> {
    private:

    public:
        DenseNet121(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet121(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    struct DenseNet169 : xt::Cloneable<DenseNet169> {
    private:

    public:
        DenseNet169(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet169(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    struct DenseNet201 : xt::Cloneable<DenseNet201> {
    private:

    public:
        DenseNet201(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet201(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    struct DenseNet264 : xt::Cloneable<DenseNet264> {
    private:

    public:
        DenseNet264(int num_classes /* classes */, int in_channels = 3/* input channels */);

        DenseNet264(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };
}
