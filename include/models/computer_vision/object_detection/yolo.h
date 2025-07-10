#pragma once
#include "../../common.h"

namespace xt::models
{
    struct Backbone : xt::Module
    {
        Backbone();
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::ReLU relu{nullptr};
    };

    // TORCH_MODULE(Backbone);

    // Sified YOLOv10 Neck
    struct Neck : xt::Module
    {
        Neck();
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
        torch::nn::Upsample upsample{nullptr};
        torch::nn::ReLU relu{nullptr};
    };

    // TORCH_MODULE(Neck);

    // Sified YOLOv10 Head
    struct Head : xt::Module
    {
        Head(int num_classes, int num_anchors = 3) ;
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        int num_classes_, num_anchors_;
        torch::nn::Conv2d conv{nullptr};
    };

    // TORCH_MODULE(Head);

    // Sified YOLOv10 Model
    struct YOLOv10 : xt::Module
    {
        YOLOv10(int num_classes, int num_anchors = 3);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x);

        std::shared_ptr<Backbone> backbone{nullptr};
        std::shared_ptr<Neck> neck{nullptr};
        std::shared_ptr<Head> head{nullptr};
    };

    // TORCH_MODULE(YOLOv10);

    // YOLO Loss Function
    struct YOLOLoss : xt::Module
    {
        YOLOLoss(int num_classes, int num_anchors, float lambda_coord = 5.0, float lambda_noobj = 0.5);

        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor pred, torch::Tensor target);

        int num_classes_, num_anchors_;
        float lambda_coord_, lambda_noobj_;
        torch::Tensor anchors_;
    };

    // TORCH_MODULE(YOLOLoss);
}
