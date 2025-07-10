#pragma once
#include "../../common.h"

namespace xt::models
{
    struct Backbone : xt::Module
    {
        Backbone()
        {
            conv1 = register_module(
                "conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1)));
            conv2 = register_module(
                "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1)));
            conv3 = register_module(
                "conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)));
            relu = register_module("relu", torch::nn::ReLU());
        }
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x)
        {
            x = relu->forward(conv1->forward(x)); // [batch, 16, 28, 28]
            x = relu->forward(conv2->forward(x)); // [batch, 32, 14, 14]
            x = relu->forward(conv3->forward(x)); // [batch, 64, 7, 7]
            return x;
        }

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
        torch::nn::ReLU relu{nullptr};
    };

    // TORCH_MODULE(Backbone);

    // Sified YOLOv10 Neck
    struct Neck : xt::Module
    {
        Neck()
        {
            conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 1).stride(1)));
            upsample = register_module(
                "upsample", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(2).mode(torch::kNearest)));
            conv2 = register_module(
                "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 3).stride(1).padding(1)));
            relu = register_module("relu", torch::nn::ReLU());
        }
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x)
        {
            x = relu->forward(conv1->forward(x)); // [batch, 32, 7, 7]
            x = upsample->forward(x); // [batch, 32, 14, 14]
            x = relu->forward(conv2->forward(x)); // [batch, 16, 14, 14]
            return x;
        }

        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
        torch::nn::Upsample upsample{nullptr};
        torch::nn::ReLU relu{nullptr};
    };

    // TORCH_MODULE(Neck);

    // Sified YOLOv10 Head
    struct Head : xt::Module
    {
        Head(int num_classes, int num_anchors = 3) : num_classes_(num_classes), num_anchors_(num_anchors)
        {
            // Output: [x, y, w, h, conf, classes] per anchor
            conv = register_module(
                "conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, num_anchors * (5 + num_classes), 1).stride(1)));
        }
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x)
        {
            x = conv->forward(x); // [batch, num_anchors * (5 + num_classes), 14, 14]
            // Reshape to [batch, num_anchors, 5 + num_classes, grid_h, grid_w]
            auto batch_size = x.size(0);
            x = x.view({batch_size, num_anchors_, 5 + num_classes_, 14, 14});
            x = x.permute({0, 1, 3, 4, 2}); // [batch, num_anchors, grid_h, grid_w, 5 + num_classes]
            // Apply sigmoid to x, y, conf, and class probs
            x.select(4, 0) = torch::sigmoid(x.select(4, 0)); // x
            x.select(4, 1) = torch::sigmoid(x.select(4, 1)); // y
            x.select(4, 4) = torch::sigmoid(x.select(4, 4)); // conf
            for (int i = 5; i < 5 + num_classes_; ++i)
            {
                x.select(4, i) = torch::sigmoid(x.select(4, i)); // class probs
            }
            return x;
        }

        int num_classes_, num_anchors_;
        torch::nn::Conv2d conv{nullptr};
    };

    // TORCH_MODULE(Head);

    // Sified YOLOv10 Model
    struct YOLOv10 : xt::Module
    {
        YOLOv10(int num_classes, int num_anchors = 3)
        {
            backbone = register_module("backbone", Backbone());
            neck = register_module("neck", Neck());
            head = register_module("head", Head(num_classes, num_anchors));
        }
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor x)
        {
            x = backbone->forward(x);
            x = neck->forward(x);
            x = head->forward(x);
            return x;
        }

        Backbone backbone{nullptr};
        Neck neck{nullptr};
        Head head{nullptr};
    };

    // TORCH_MODULE(YOLOv10);

    // YOLO Loss Function
    struct YOLOLoss : xt::Module
    {
        YOLOLoss(int num_classes, int num_anchors, float lambda_coord = 5.0, float lambda_noobj = 0.5)
            : num_classes_(num_classes), num_anchors_(num_anchors), lambda_coord_(lambda_coord),
              lambda_noobj_(lambda_noobj)
        {
            // Predefined anchors (width, height) scaled to grid
            anchors_ = torch::tensor({{1.0, 1.0}, {2.0, 2.0}, {0.5, 0.5}}, torch::kFloat32);
        }
        auto forward(std::initializer_list <std::any> tensors) -> std::any override;

        torch::Tensor forward(torch::Tensor pred, torch::Tensor target)
        {
            auto batch_size = pred.size(0);
            auto grid_size = pred.size(2); // 14x14
            auto device = pred.device();

            // Initialize losses
            torch::Tensor loss_xy = torch::zeros({}, device);
            torch::Tensor loss_wh = torch::zeros({}, device);
            torch::Tensor loss_conf = torch::zeros({}, device);
            torch::Tensor loss_class = torch::zeros({}, device);

            // Generate grid offsets
            auto grid_x = torch::arange(grid_size, torch::kFloat32, device).repeat({grid_size, 1});
            auto grid_y = grid_x.transpose(0, 1);
            grid_x = grid_x.view({1, 1, grid_size, grid_size, 1});
            grid_y = grid_y.view({1, 1, grid_size, grid_size, 1});

            // Process each batch
            for (int b = 0; b < batch_size; ++b)
            {
                auto pred_b = pred[b]; // [num_anchors, grid_h, grid_w, 5 + num_classes]
                auto target_b = target[b]; // [max_objects, 5] (x, y, w, h, class)

                // Create target tensors
                auto obj_mask = torch::zeros({num_anchors_, grid_size, grid_size}, torch::kBool, device);
                auto noobj_mask = torch::ones({num_anchors_, grid_size, grid_size}, torch::kBool, device);
                auto target_xy = torch::zeros({num_anchors_, grid_size, grid_size, 2}, torch::kFloat32, device);
                auto target_wh = torch::zeros({num_anchors_, grid_size, grid_size, 2}, torch::kFloat32, device);
                auto target_conf = torch::zeros({num_anchors_, grid_size, grid_size}, torch::kFloat32, device);
                auto target_class = torch::zeros({num_anchors_, grid_size, grid_size, num_classes_}, torch::kFloat32,
                                                 device);

                // Assign ground truth to grid cells
                for (int t = 0; t < target_b.size(0); ++t)
                {
                    if (target_b[t][4].item<float>() < 0) continue; // Invalid object
                    float tx = target_b[t][0].item<float>() * grid_size;
                    float ty = target_b[t][1].item<float>() * grid_size;
                    float tw = target_b[t][2].item<float>() * grid_size;
                    float th = target_b[t][3].item<float>() * grid_size;
                    int class_id = target_b[t][4].item<float>();
                    int gx = static_cast<int>(tx);
                    int gy = static_cast<int>(ty);
                    if (gx >= grid_size || gy >= grid_size || gx < 0 || gy < 0) continue;

                    // Compute IoU with anchors to select best anchor
                    auto anchor_boxes = anchors_.clone().to(device);
                    auto gt_box = torch::tensor({tw, th}, torch::kFloat32, device).unsqueeze(0);
                    auto anchor_w = anchor_boxes.select(1, 0);
                    auto anchor_h = anchor_boxes.select(1, 1);
                    auto inter_w = torch::min(anchor_w, gt_box[0][0]);
                    auto inter_h = torch::min(anchor_h, gt_box[0][1]);
                    auto inter_area = inter_w * inter_h;
                    auto union_area = anchor_w * anchor_h + gt_box[0][0] * gt_box[0][1] - inter_area;
                    auto iou = inter_area / (union_area + 1e-6);
                    auto [_, best_anchor] = torch::max(iou, 0);

                    // Assign target
                    obj_mask[best_anchor.item<int>()][gy][gx] = true;
                    noobj_mask[best_anchor.item<int>()][gy][gx] = false;
                    target_xy[best_anchor.item<int>()][gy][gx] = torch::tensor(
                        {tx - gx, ty - gy}, torch::kFloat32, device);
                    target_wh[best_anchor.item<int>()][gy][gx] = torch::log(torch::tensor({
                            tw / anchor_boxes[best_anchor][0].item<float>(),
                            th / anchor_boxes[best_anchor][1].item<float>()
                        },
                        torch::kFloat32, device) + 1e-6);
                    target_conf[best_anchor.item<int>()][gy][gx] = 1.0;
                    target_class[best_anchor.item<int>()][gy][gx][class_id] = 1.0;
                }

                // Compute losses
                auto pred_xy = pred_b.slice(4, 0, 2); // [num_anchors, grid_h, grid_w, 2]
                auto pred_wh = pred_b.slice(4, 2, 4); // [num_anchors, grid_h, grid_w, 2]
                auto pred_conf = pred_b.select(4, 4); // [num_anchors, grid_h, grid_w]
                auto pred_class = pred_b.slice(4, 5, 5 + num_classes_); // [num_anchors, grid_h, grid_w, num_classes]

                loss_xy += lambda_coord_ * torch::nn::functional::mse_loss(pred_xy.masked_select(obj_mask.unsqueeze(3)),
                                                                           target_xy.masked_select(
                                                                               obj_mask.unsqueeze(3)));
                loss_wh += lambda_coord_ * torch::nn::functional::mse_loss(pred_wh.masked_select(obj_mask.unsqueeze(3)),
                                                                           target_wh.masked_select(
                                                                               obj_mask.unsqueeze(3)));
                loss_conf += torch::nn::functional::binary_cross_entropy(
                        pred_conf.masked_select(obj_mask), target_conf.masked_select(obj_mask)) +
                    lambda_noobj_ * torch::nn::functional::binary_cross_entropy(
                        pred_conf.masked_select(noobj_mask), target_conf.masked_select(noobj_mask));
                loss_class += torch::nn::functional::binary_cross_entropy(
                    pred_class.masked_select(obj_mask.unsqueeze(3)),
                    target_class.masked_select(obj_mask.unsqueeze(3)));
            }

            return (loss_xy + loss_wh + loss_conf + loss_class) / batch_size;
        }

        int num_classes_, num_anchors_;
        float lambda_coord_, lambda_noobj_;
        torch::Tensor anchors_;
    };

    // TORCH_MODULE(YOLOLoss);
}
