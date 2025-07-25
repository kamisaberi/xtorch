#include <models/computer_vision/image_classification/resnet.h>

namespace xt::models
{
    namespace
    {
        ResidualBlock::ResidualBlock(int in_channels, int out_channels, int stride, torch::nn::Sequential downsample)
        {
            conv1 = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)),
                torch::nn::BatchNorm2d(out_channels),
                torch::nn::ReLU()
            );

            conv2 = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1)),
                torch::nn::BatchNorm2d(out_channels)
            );

            register_module("conv1", conv1);
            register_module("conv2", conv2);


            this->downsample = downsample;
            this->relu = torch::nn::ReLU();
            this->out_channels = out_channels;
        }

        auto ResidualBlock::forward(std::initializer_list<std::any> tensors) -> std::any
        {
            std::vector<std::any> any_vec(tensors);

            std::vector<torch::Tensor> tensor_vec;
            for (const auto& item : any_vec)
            {
                tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
            }

            torch::Tensor x = tensor_vec[0];

            residual = x;
            torch::Tensor out = conv1->forward(x);
            out = conv2->forward(out);
            if (downsample)
            {
                residual = downsample->forward(x);
            }
            else
            {
            }
            out += residual;
            out = relu(out);
            return out;
        }
    }


    ResNet18::ResNet18(vector<int> layers, int num_classes, int in_channels)
    {
        inplanes = 64;


        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );

        register_module("conv1", conv1);

        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        // maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        fc = torch::nn::Linear(512, num_classes);
        reset();
    }


    ResNet18::ResNet18(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
        inplanes = 64;

        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("conv1", conv1);
        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        // Compute flattened size dynamically
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = conv1->forward(dummy_input);
        x = maxpool->forward(x);
        x = layer0->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);

        // Adaptive pooling instead of fixed 7x7 pooling
        auto spatial_dims = x.sizes().slice(2); // Get H and W after all convs
        // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        x = avgpool->forward(x);
        int flattened_size = x.numel() / x.size(0);

        fc = torch::nn::Linear(flattened_size, num_classes);

        register_module("avgpool", avgpool);
        register_module("fc", fc);
        reset();
    }

    // torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    //     torch::nn::Sequential downsample = nullptr;
    //     if (stride != 1 || inplanes != planes) {
    //         downsample = torch::nn::Sequential();
    //         downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)));
    //         downsample->push_back(torch::nn::BatchNorm2d(planes));
    //     }
    //
    //     auto layers = torch::nn::Sequential();
    //     layers->push_back(ResidualBlock(inplanes, planes, stride, downsample));
    //     inplanes = planes;
    //     for (int i = 1; i < blocks; ++i) {
    //         layers->push_back(ResidualBlock(inplanes, planes));
    //     }
    //     return layers;
    // }


    torch::nn::Sequential ResNet18::makeLayerFromResidualBlock(int planes, int blocks, int stride)
    {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes)
        {
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
        // layers->push_back(rb);
        // inplanes = planes;
        // for (int i = 1; i < blocks; i++)
        // {
        //     ResidualBlock rbt = ResidualBlock(inplanes, planes);
        //     layers->push_back(rbt);
        // }
        return layers;
    }

    auto ResNet18::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
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

    void ResNet18::reset()
    {
    }


    ResNet34::ResNet34(vector<int> layers, int num_classes, int in_channels)
    {
        inplanes = 64;


        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );

        register_module("conv1", conv1);

        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        // maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        fc = torch::nn::Linear(512, num_classes);
        reset();
    }


    ResNet34::ResNet34(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape)

    {
        inplanes = 64;

        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("conv1", conv1);
        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        // Compute flattened size dynamically
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = conv1->forward(dummy_input);
        x = maxpool->forward(x);
        x = layer0->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);

        // Adaptive pooling instead of fixed 7x7 pooling
        auto spatial_dims = x.sizes().slice(2); // Get H and W after all convs
        // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        x = avgpool->forward(x);
        int flattened_size = x.numel() / x.size(0);

        fc = torch::nn::Linear(flattened_size, num_classes);

        register_module("avgpool", avgpool);
        register_module("fc", fc);
        reset();
    }

    // torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    //     torch::nn::Sequential downsample = nullptr;
    //     if (stride != 1 || inplanes != planes) {
    //         downsample = torch::nn::Sequential();
    //         downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)));
    //         downsample->push_back(torch::nn::BatchNorm2d(planes));
    //     }
    //
    //     auto layers = torch::nn::Sequential();
    //     layers->push_back(ResidualBlock(inplanes, planes, stride, downsample));
    //     inplanes = planes;
    //     for (int i = 1; i < blocks; ++i) {
    //         layers->push_back(ResidualBlock(inplanes, planes));
    //     }
    //     return layers;
    // }


    torch::nn::Sequential ResNet34::makeLayerFromResidualBlock(int planes, int blocks, int stride)
    {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes)
        {
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
        // layers->push_back(rb);
        // inplanes = planes;
        // for (int i = 1; i < blocks; i++)
        // {
        //     ResidualBlock rbt = ResidualBlock(inplanes, planes);
        //     layers->push_back(rbt);
        // }
        return layers;
    }

    auto ResNet34::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
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

    void ResNet34::reset()
    {
    }


    ResNet50::ResNet50(vector<int> layers, int num_classes, int in_channels)
    {
        inplanes = 64;


        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );

        register_module("conv1", conv1);

        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        // maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        fc = torch::nn::Linear(512, num_classes);
        reset();
    }


    ResNet50::ResNet50(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape)

    {
        inplanes = 64;

        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("conv1", conv1);
        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        // Compute flattened size dynamically
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = conv1->forward(dummy_input);
        x = maxpool->forward(x);
        x = layer0->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);

        // Adaptive pooling instead of fixed 7x7 pooling
        auto spatial_dims = x.sizes().slice(2); // Get H and W after all convs
        // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        x = avgpool->forward(x);
        int flattened_size = x.numel() / x.size(0);

        fc = torch::nn::Linear(flattened_size, num_classes);

        register_module("avgpool", avgpool);
        register_module("fc", fc);
        reset();
    }

    // torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    //     torch::nn::Sequential downsample = nullptr;
    //     if (stride != 1 || inplanes != planes) {
    //         downsample = torch::nn::Sequential();
    //         downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)));
    //         downsample->push_back(torch::nn::BatchNorm2d(planes));
    //     }
    //
    //     auto layers = torch::nn::Sequential();
    //     layers->push_back(ResidualBlock(inplanes, planes, stride, downsample));
    //     inplanes = planes;
    //     for (int i = 1; i < blocks; ++i) {
    //         layers->push_back(ResidualBlock(inplanes, planes));
    //     }
    //     return layers;
    // }


    torch::nn::Sequential ResNet50::makeLayerFromResidualBlock(int planes, int blocks, int stride)
    {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes)
        {
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
        // layers->push_back(rb);
        // inplanes = planes;
        // for (int i = 1; i < blocks; i++)
        // {
        //     ResidualBlock rbt = ResidualBlock(inplanes, planes);
        //     layers->push_back(rbt);
        // }
        return layers;
    }

    auto ResNet50::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
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

    void ResNet50::reset()
    {
    }


    ResNet101::ResNet101(vector<int> layers, int num_classes, int in_channels)
    {
        inplanes = 64;


        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );

        register_module("conv1", conv1);

        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        // maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        fc = torch::nn::Linear(512, num_classes);
        reset();
    }


    ResNet101::ResNet101(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape)

    {
        inplanes = 64;

        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("conv1", conv1);
        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        // Compute flattened size dynamically
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = conv1->forward(dummy_input);
        x = maxpool->forward(x);
        x = layer0->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);

        // Adaptive pooling instead of fixed 7x7 pooling
        auto spatial_dims = x.sizes().slice(2); // Get H and W after all convs
        // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        x = avgpool->forward(x);
        int flattened_size = x.numel() / x.size(0);

        fc = torch::nn::Linear(flattened_size, num_classes);

        register_module("avgpool", avgpool);
        register_module("fc", fc);
        reset();
    }

    // torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    //     torch::nn::Sequential downsample = nullptr;
    //     if (stride != 1 || inplanes != planes) {
    //         downsample = torch::nn::Sequential();
    //         downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)));
    //         downsample->push_back(torch::nn::BatchNorm2d(planes));
    //     }
    //
    //     auto layers = torch::nn::Sequential();
    //     layers->push_back(ResidualBlock(inplanes, planes, stride, downsample));
    //     inplanes = planes;
    //     for (int i = 1; i < blocks; ++i) {
    //         layers->push_back(ResidualBlock(inplanes, planes));
    //     }
    //     return layers;
    // }


    torch::nn::Sequential ResNet101::makeLayerFromResidualBlock(int planes, int blocks, int stride)
    {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes)
        {
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
        // layers->push_back(rb);
        // inplanes = planes;
        // for (int i = 1; i < blocks; i++)
        // {
        //     ResidualBlock rbt = ResidualBlock(inplanes, planes);
        //     layers->push_back(rbt);
        // }
        return layers;
    }

    auto ResNet101::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
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

    void ResNet101::reset()
    {
    }


    ResNet152::ResNet152(vector<int> layers, int num_classes, int in_channels)
    {
        inplanes = 64;


        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );

        register_module("conv1", conv1);

        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        // maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        fc = torch::nn::Linear(512, num_classes);
        reset();
    }


    ResNet152::ResNet152(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape)

    {
        inplanes = 64;

        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("conv1", conv1);
        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        // Compute flattened size dynamically
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = conv1->forward(dummy_input);
        x = maxpool->forward(x);
        x = layer0->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);

        // Adaptive pooling instead of fixed 7x7 pooling
        auto spatial_dims = x.sizes().slice(2); // Get H and W after all convs
        // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        x = avgpool->forward(x);
        int flattened_size = x.numel() / x.size(0);

        fc = torch::nn::Linear(flattened_size, num_classes);

        register_module("avgpool", avgpool);
        register_module("fc", fc);
        reset();
    }

    // torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    //     torch::nn::Sequential downsample = nullptr;
    //     if (stride != 1 || inplanes != planes) {
    //         downsample = torch::nn::Sequential();
    //         downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)));
    //         downsample->push_back(torch::nn::BatchNorm2d(planes));
    //     }
    //
    //     auto layers = torch::nn::Sequential();
    //     layers->push_back(ResidualBlock(inplanes, planes, stride, downsample));
    //     inplanes = planes;
    //     for (int i = 1; i < blocks; ++i) {
    //         layers->push_back(ResidualBlock(inplanes, planes));
    //     }
    //     return layers;
    // }


    torch::nn::Sequential ResNet152::makeLayerFromResidualBlock(int planes, int blocks, int stride)
    {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes)
        {
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
        // layers->push_back(rb);
        // inplanes = planes;
        // for (int i = 1; i < blocks; i++)
        // {
        //     ResidualBlock rbt = ResidualBlock(inplanes, planes);
        //     layers->push_back(rbt);
        // }
        return layers;
    }

    auto ResNet152::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
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

    void ResNet152::reset()
    {
    }

    ResNet200::ResNet200(vector<int> layers, int num_classes, int in_channels)
    {
        inplanes = 64;


        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );

        register_module("conv1", conv1);

        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        // maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        fc = torch::nn::Linear(512, num_classes);
        reset();
    }


    ResNet200::ResNet200(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
        inplanes = 64;

        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("conv1", conv1);
        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        // Compute flattened size dynamically
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = conv1->forward(dummy_input);
        x = maxpool->forward(x);
        x = layer0->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);

        // Adaptive pooling instead of fixed 7x7 pooling
        auto spatial_dims = x.sizes().slice(2); // Get H and W after all convs
        // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        x = avgpool->forward(x);
        int flattened_size = x.numel() / x.size(0);

        fc = torch::nn::Linear(flattened_size, num_classes);

        register_module("avgpool", avgpool);
        register_module("fc", fc);
        reset();
    }

    // torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    //     torch::nn::Sequential downsample = nullptr;
    //     if (stride != 1 || inplanes != planes) {
    //         downsample = torch::nn::Sequential();
    //         downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)));
    //         downsample->push_back(torch::nn::BatchNorm2d(planes));
    //     }
    //
    //     auto layers = torch::nn::Sequential();
    //     layers->push_back(ResidualBlock(inplanes, planes, stride, downsample));
    //     inplanes = planes;
    //     for (int i = 1; i < blocks; ++i) {
    //         layers->push_back(ResidualBlock(inplanes, planes));
    //     }
    //     return layers;
    // }


    torch::nn::Sequential ResNet200::makeLayerFromResidualBlock(int planes, int blocks, int stride)
    {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes)
        {
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
        // layers->push_back(rb);
        // inplanes = planes;
        // for (int i = 1; i < blocks; i++)
        // {
        //     ResidualBlock rbt = ResidualBlock(inplanes, planes);
        //     layers->push_back(rbt);
        // }
        return layers;
    }

    auto ResNet200::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
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

    void ResNet200::reset()
    {
    }

    ResNet1202::ResNet1202(vector<int> layers, int num_classes, int in_channels)
    {
        inplanes = 64;


        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );

        register_module("conv1", conv1);

        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));
        // maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        fc = torch::nn::Linear(512, num_classes);
        reset();
    }


    ResNet1202::ResNet1202(std::vector<int> layers, int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
        inplanes = 64;

        conv1 = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, 64, 7).stride(2).padding(3)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU()
        );
        maxpool = torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1));

        layer0 = makeLayerFromResidualBlock(64, layers[0], 1);
        layer1 = makeLayerFromResidualBlock(128, layers[1], 2);
        layer2 = makeLayerFromResidualBlock(256, layers[2], 2);
        layer3 = makeLayerFromResidualBlock(512, layers[3], 2);

        register_module("conv1", conv1);
        register_module("layer0", layer0);
        register_module("layer1", layer1);
        register_module("layer2", layer2);
        register_module("layer3", layer3);

        // Compute flattened size dynamically
        torch::NoGradGuard no_grad;
        auto dummy_input = torch::zeros({1, in_channels, input_shape[0], input_shape[1]});
        auto x = conv1->forward(dummy_input);
        x = maxpool->forward(x);
        x = layer0->forward(x);
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);

        // Adaptive pooling instead of fixed 7x7 pooling
        auto spatial_dims = x.sizes().slice(2); // Get H and W after all convs
        // avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
        avgpool = torch::nn::AvgPool2d(torch::nn::AvgPool2dOptions(7).stride(1));
        x = avgpool->forward(x);
        int flattened_size = x.numel() / x.size(0);

        fc = torch::nn::Linear(flattened_size, num_classes);

        register_module("avgpool", avgpool);
        register_module("fc", fc);
        reset();
    }

    // torch::nn::Sequential ResNet::makeLayerFromResidualBlock(int planes, int blocks, int stride) {
    //     torch::nn::Sequential downsample = nullptr;
    //     if (stride != 1 || inplanes != planes) {
    //         downsample = torch::nn::Sequential();
    //         downsample->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(inplanes, planes, 1).stride(stride)));
    //         downsample->push_back(torch::nn::BatchNorm2d(planes));
    //     }
    //
    //     auto layers = torch::nn::Sequential();
    //     layers->push_back(ResidualBlock(inplanes, planes, stride, downsample));
    //     inplanes = planes;
    //     for (int i = 1; i < blocks; ++i) {
    //         layers->push_back(ResidualBlock(inplanes, planes));
    //     }
    //     return layers;
    // }


    torch::nn::Sequential ResNet1202::makeLayerFromResidualBlock(int planes, int blocks, int stride)
    {
        torch::nn::Sequential downsample = nullptr;
        if (stride != 1 || inplanes != planes)
        {
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
        // layers->push_back(rb);
        // inplanes = planes;
        // for (int i = 1; i < blocks; i++)
        // {
        //     ResidualBlock rbt = ResidualBlock(inplanes, planes);
        //     layers->push_back(rbt);
        // }
        return layers;
    }

    auto ResNet1202::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
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


    void ResNet1202::reset()
    {
    }
}
