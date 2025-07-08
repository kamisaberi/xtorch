#pragma once

#include "../../common.h"


using namespace std;


namespace xt::models {

    // Swish activation (x * sigmoid(x))
    torch::Tensor swish(torch::Tensor x);

    // Squeeze-and-Excitation Block
    struct SEBlockImpl : torch::nn::Module {
        SEBlockImpl(int in_channels, int reduction);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    };

    TORCH_MODULE(SEBlock);

    // MBConv Block (Inverted Residual with Depthwise Separable Conv)
    struct MBConvBlockImpl : torch::nn::Module {
        MBConvBlockImpl(int in_channels, int out_channels, int expansion, int kernel_size, int stride, int reduction);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d expand_conv{nullptr}, depthwise_conv{nullptr}, pointwise_conv{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr}, bn2{nullptr};
        SEBlock se{nullptr};
        bool skip_connection;
    };

    TORCH_MODULE(MBConvBlock);

    // EfficientNetB0
    struct EfficientNetB0Impl : torch::nn::Module {
        EfficientNetB0Impl(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
        torch::nn::Linear fc{nullptr};
        torch::nn::ModuleList blocks{torch::nn::ModuleList()};
    };

    TORCH_MODULE(EfficientNetB0);


    struct EfficientNetB0 : xt::Cloneable<EfficientNetB0> {
    private:

    public:
        EfficientNetB0(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB0(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // EfficientNetB1
    struct EfficientNetB1Impl : torch::nn::Module {
        EfficientNetB1Impl(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
        torch::nn::Linear fc{nullptr};
        torch::nn::ModuleList blocks{torch::nn::ModuleList()};
    };

    TORCH_MODULE(EfficientNetB1);


    struct EfficientNetB1 : xt::Cloneable<EfficientNetB1> {
    private:

    public:
        EfficientNetB1(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB1(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // EfficientNetB2
    struct EfficientNetB2Impl : torch::nn::Module {
        EfficientNetB2Impl(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
        torch::nn::Linear fc{nullptr};
        torch::nn::ModuleList blocks{torch::nn::ModuleList()};
    };

    TORCH_MODULE(EfficientNetB2);


    struct EfficientNetB2 : xt::Cloneable<EfficientNetB2> {
    private:

    public:
        EfficientNetB2(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB2(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // EfficientNetB3
    struct EfficientNetB3Impl : torch::nn::Module {
        EfficientNetB3Impl(int num_classes = 10);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
        torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
        torch::nn::Linear fc{nullptr};
        torch::nn::ModuleList blocks{torch::nn::ModuleList()};
    };

    TORCH_MODULE(EfficientNetB3);


    struct EfficientNetB3 : xt::Cloneable<EfficientNetB3> {
    private:

    public:
        EfficientNetB3(int num_classes /* classes */, int in_channels = 3/* input channels */);

        EfficientNetB3(int num_classes, int in_channels, std::vector <int64_t> input_shape);

        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        void reset() override;
    };


    // EfficientNetB4
    EfficientNetB4Impl(int num_classes = 10);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
    torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::ModuleList blocks{torch::nn::ModuleList()};
};
TORCH_MODULE(EfficientNetB4);

struct EfficientNetB4 : xt::Cloneable<EfficientNetB4> {
private:

public:
    EfficientNetB4(int num_classes /* classes */, int in_channels = 3/* input channels */);

    EfficientNetB4(int num_classes, int in_channels, std::vector <int64_t> input_shape);

    auto forward(std::initializer_list <std::any> tensors) -> std::any override;

    void reset() override;
};


// EfficientNetB5
struct EfficientNetB5Impl : torch::nn::Module {
    EfficientNetB5Impl(int num_classes = 10);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
    torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::ModuleList blocks{torch::nn::ModuleList()};
};

TORCH_MODULE(EfficientNetB5);

struct EfficientNetB5 : xt::Cloneable<EfficientNetB5> {
private:

public:
    EfficientNetB5(int num_classes /* classes */, int in_channels = 3/* input channels */);

    EfficientNetB5(int num_classes, int in_channels, std::vector <int64_t> input_shape);

    auto forward(std::initializer_list <std::any> tensors) -> std::any override;

    void reset() override;
};


// EfficientNetB6
struct EfficientNetB6Impl : torch::nn::Module {
    EfficientNetB6Impl(int num_classes = 10);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
    torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::ModuleList blocks{torch::nn::ModuleList()};
};

TORCH_MODULE(EfficientNetB6);


struct EfficientNetB6 : xt::Cloneable<EfficientNetB6> {
private:

public:
    EfficientNetB6(int num_classes /* classes */, int in_channels = 3/* input channels */);

    EfficientNetB6(int num_classes, int in_channels, std::vector <int64_t> input_shape);

    auto forward(std::initializer_list <std::any> tensors) -> std::any override;

    void reset() override;
};


// EfficientNetB7
struct EfficientNetB7Impl : torch::nn::Module {
    EfficientNetB7Impl(int num_classes = 10);

    torch::Tensor forward(torch::Tensor x);

    torch::nn::Conv2d stem_conv{nullptr}, head_conv{nullptr};
    torch::nn::BatchNorm2d bn0{nullptr}, bn1{nullptr};
    torch::nn::Linear fc{nullptr};
    torch::nn::ModuleList blocks{torch::nn::ModuleList()};
};

TORCH_MODULE(EfficientNetB7);

struct EfficientNetB7 : xt::Cloneable<EfficientNetB7> {
private:

public:
    EfficientNetB7(int num_classes /* classes */, int in_channels = 3/* input channels */);

    EfficientNetB7(int num_classes, int in_channels, std::vector <int64_t> input_shape);

    auto forward(std::initializer_list <std::any> tensors) -> std::any override;

    void reset() override;
};

}
