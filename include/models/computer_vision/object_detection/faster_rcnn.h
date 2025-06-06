#pragma once
#include "../../common.h"


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // COCO class names (91 classes, including background)
// const std::vector<std::string> COCO_CLASSES = {
//     "background", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
//     "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
//     "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//     "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
//     "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
//     "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
//     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
//     "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
//     "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
//     "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//     "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
//     "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
// };
//
// // Backbone: Simplified ResNet-like architecture
// struct BackboneImpl : torch::nn::Module {
//     BackboneImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//         maxpool = register_module("maxpool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
//
//         // Simplified residual layers
//         layer1 = register_module("layer1", make_layer(64, 64, 2));
//         layer2 = register_module("layer2", make_layer(64, 128, 2, 2));
//     }
//
//     torch::nn::Sequential make_layer(int64_t in_channels, int64_t out_channels, int64_t blocks, int64_t stride = 1) {
//         torch::nn::Sequential layers;
//         layers->push_back(torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, out_channels, 3).stride(stride).padding(1)));
//         layers->push_back(torch::nn::BatchNorm2d(out_channels));
//         layers->push_back(torch::nn::ReLU());
//         for (int64_t i = 0; i < blocks - 1; ++i) {
//             layers->push_back(torch::nn::Conv2d(
//                 torch::nn::Conv2dOptions(out_channels, out_channels, 3).stride(1).padding(1)));
//             layers->push_back(torch::nn::BatchNorm2d(out_channels));
//             layers->push_back(torch::nn::ReLU());
//         }
//         return layers;
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = conv1->forward(x);
//         x = bn1->forward(x);
//         x = relu->forward(x);
//         x = maxpool->forward(x);
//         x = layer1->forward(x);
//         x = layer2->forward(x);
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::MaxPool2d maxpool{nullptr};
//     torch::nn::Sequential layer1{nullptr}, layer2{nullptr};
// };
// TORCH_MODULE(Backbone);
//
// // Region Proposal Network (RPN)
// struct RPNImpl : torch::nn::Module {
//     RPNImpl(int64_t in_channels, int64_t mid_channels = 256, int64_t num_anchors = 9) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, mid_channels, 3).stride(1).padding(1)));
//         cls_logits = register_module("cls_logits", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(mid_channels, num_anchors * 2, 1)));
//         bbox_pred = register_module("bbox_pred", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(mid_channels, num_anchors * 4, 1)));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         x = torch::relu(conv->forward(x));
//         auto cls = cls_logits->forward(x); // Objectness scores
//         auto bbox = bbox_pred->forward(x); // Bounding box deltas
//         return std::make_tuple(cls, bbox);
//     }
//
//     torch::nn::Conv2d conv{nullptr}, cls_logits{nullptr}, bbox_pred{nullptr};
// };
// TORCH_MODULE(RPN);
//
// // Detection Head
// struct DetectionHeadImpl : torch::nn::Module {
//     DetectionHeadImpl(int64_t in_channels, int64_t num_classes) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, 1024));
//         fc2 = register_module("fc2", torch::nn::Linear(1024, 1024));
//         cls_score = register_module("cls_score", torch::nn::Linear(1024, num_classes));
//         bbox_pred = register_module("bbox_pred", torch::nn::Linear(1024, num_classes * 4));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         x = torch::relu(fc1->forward(x));
//         x = torch::relu(fc2->forward(x));
//         auto cls = cls_score->forward(x);
//         auto bbox = bbox_pred->forward(x);
//         return std::make_tuple(cls, bbox);
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr}, cls_score{nullptr}, bbox_pred{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(DetectionHead);
//
// // Faster R-CNN Model
// struct FasterRCNNImpl : torch::nn::Module {
//     FasterRCNNImpl(int64_t num_classes) {
//         backbone = register_module("backbone", Backbone());
//         rpn = register_module("rpn", RPN(128, 256, 9)); // 128 from backbone output
//         head = register_module("head", DetectionHead(128, num_classes));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // Backbone
//         auto features = backbone->forward(x);
//
//         // RPN
//         auto [rpn_cls, rpn_bbox] = rpn->forward(features);
//
//         // Simulate ROI pooling (simplified, assumes fixed-size proposals)
//         auto rois = torch::randn({100, 4}); // Dummy ROIs for example
//         auto pooled_features = torch::adaptive_avg_pool2d(features, {7, 7});
//         pooled_features = pooled_features.view({pooled_features.size(0), -1});
//
//         // Detection Head
//         auto [cls_scores, bbox_deltas] = head->forward(pooled_features);
//
//         return std::make_tuple(cls_scores, bbox_deltas, rpn_cls);
//     }
//
//     Backbone backbone{nullptr};
//     RPN rpn{nullptr};
//     DetectionHead head{nullptr};
// };
// TORCH_MODULE(FasterRCNN);
//
// // Main function to test the model
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Initialize model
//         FasterRCNN model(91); // 91 classes for COCO
//         model->to(device);
//         model->eval();
//
//         // Load and preprocess image
//         cv::Mat image = cv::imread("input_image.jpg");
//         if (image.empty()) {
//             std::cerr << "Error: Could not load image." << std::endl;
//             return -1;
//         }
//
//         cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//         image.convertTo(image, CV_32F, 1.0 / 255.0);
//
//         torch::Tensor img_tensor = torch::from_blob(
//             image.data, {1, image.rows, image.cols, 3}, torch::kFloat32
//         );
//         img_tensor = img_tensor.permute({0, 3, 1, 2});
//         img_tensor = img_tensor.to(device);
//
//         // Forward pass
//         auto [cls_scores, bbox_deltas, rpn_cls] = model->forward(img_tensor);
//
//         // Post-process (simplified)
//         cls_scores = torch::softmax(cls_scores, 1);
//         auto max_scores = std::get<1>(torch::max(cls_scores, 1));
//         for (int i = 0; i < max_scores.size(0); ++i) {
//             float score = max_scores[i].item<float>();
//             if (score > 0.5) {
//                 int label = cls_scores[i].argmax().item<int>();
//                 std::cout << "Detected: " << COCO_CLASSES[label] << " with confidence " << score << std::endl;
//             }
//         }
//
//         std::cout << "Inference complete." << std::endl;
//
//     } catch (const std::exception& e) {
//         std::cerr << "Error: " << e.what() << std::endl;
//         return -1;
//     }
//
//     return 0;
// }

namespace xt::models
{
    struct FasterRCNN : xt::Cloneable<FasterRCNN>
    {
    private:

    public:
        FasterRCNN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        FasterRCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}