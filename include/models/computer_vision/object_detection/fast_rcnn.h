#pragma once
#include "../../common.h"


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <nlohmann/json.hpp>
// #include <fstream>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Reduced class set for smaller dataset
// const std::vector<std::string> CLASSES = {"background", "person", "car", "dog"};
// const int NUM_CLASSES = CLASSES.size(); // 4 (including background)
//
// // COCO-style dataset loader
// struct COCODataset : torch::data::Dataset<COCODataset> {
//     COCODataset(const std::string& img_dir, const std::string& ann_file) {
//         std::ifstream file(ann_file);
//         nlohmann::json json;
//         file >> json;
//
//         for (const auto& img : json["images"]) {
//             image_paths.push_back(img_dir + "/" + img["file_name"].get<std::string>());
//             image_ids.push_back(img["id"].get<int>());
//         }
//
//         for (const auto& ann : json["annotations"]) {
//             annotations[ann["image_id"].get<int>()].push_back({
//                 ann["bbox"].get<std::vector<float>>(),
//                 ann["category_id"].get<int>()
//             });
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         // Load image
//         cv::Mat image = cv::imread(image_paths[index]);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths[index]);
//         }
//         cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//         image.convertTo(image, CV_32F, 1.0 / 255.0);
//
//         torch::Tensor img_tensor = torch::from_blob(
//             image.data, {image.rows, image.cols, 3}, torch::kFloat32
//         ).permute({2, 0, 1}); // [C, H, W]
//
//         // Load annotations
//         auto anns = annotations[image_ids[index]];
//         std::vector<float> boxes;
//         std::vector<int64_t> labels;
//         for (const auto& ann : anns) {
//             boxes.push_back(ann.bbox[0]); // x1
//             boxes.push_back(ann.bbox[1]); // y1
//             boxes.push_back(ann.bbox[0] + ann.bbox[2]); // x2
//             boxes.push_back(ann.bbox[1] + ann.bbox[3]); // y2
//             labels.push_back(ann.category_id);
//         }
//
//         torch::Tensor box_tensor = torch::tensor(boxes).reshape({-1, 4});
//         torch::Tensor label_tensor = torch::tensor(labels, torch::kInt64);
//         torch::Dict<std::string, torch::Tensor> target_dict;
//         target_dict.insert("boxes", box_tensor);
//         target_dict.insert("labels", label_tensor);
//
//         return {img_tensor, target_dict};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths.size();
//     }
//
//     struct Annotation {
//         std::vector<float> bbox; // [x, y, w, h]
//         int category_id;
//     };
//
//     std::vector<std::string> image_paths;
//     std::vector<int> image_ids;
//     std::map<int, std::vector<Annotation>> annotations;
// };
//
// // Simplified ResNet Backbone
// struct BackboneImpl : torch::nn::Module {
//     BackboneImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//         maxpool = register_module("maxpool", torch::nn::MaxPool2d(
//             torch::nn::MaxPool2dOptions(3).stride(2).padding(1)));
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
//         auto cls = cls_logits->forward(x);
//         auto bbox = bbox_pred->forward(x);
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
//         rpn = register_module("rpn", RPN(128, 256, 9));
//         head = register_module("head", DetectionHead(128 * 7 * 7, num_classes));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(
//         torch::Tensor x,
//         const std::vector<torch::Dict<std::string, torch::Tensor>>& targets = {}) {
//         auto features = backbone->forward(x);
//         auto [rpn_cls, rpn_bbox] = rpn->forward(features);
//
//         // Simplified ROI pooling
//         auto pooled = torch::adaptive_avg_pool2d(features, {7, 7}).view({-1, 128 * 7 * 7});
//         auto [cls_scores, bbox_deltas] = head->forward(pooled);
//
//         if (!targets.empty()) {
//             // Compute losses during training
//             auto cls_loss = torch::nn::functional::cross_entropy(
//                 cls_scores, targets[0].at("labels"));
//             auto bbox_loss = torch::nn::functional::smooth_l1_loss(
//                 bbox_deltas, targets[0].at("boxes").view({-1, NUM_CLASSES * 4}));
//             auto rpn_cls_loss = torch::nn::functional::cross_entropy(
//                 rpn_cls.view({-1, 2}), torch::ones({rpn_cls.numel() / 2}, torch::kInt64));
//             return std::make_tuple(cls_loss + bbox_loss + rpn_cls_loss, cls_scores, bbox_deltas);
//         }
//
//         return std::make_tuple(torch::Tensor(), cls_scores, bbox_deltas);
//     }
//
//     Backbone backbone{nullptr};
//     RPN rpn{nullptr};
//     DetectionHead head{nullptr};
// };
// TORCH_MODULE(FasterRCNN);
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Initialize model and optimizer
//         FasterRCNN model(NUM_CLASSES);
//         model->to(device);
//         torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.001).momentum(0.9));
//
//         // Load dataset
//         auto dataset = COCODataset("./data/images", "./data/annotations.json")
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(2).workers(2));
//
//         // Training loop
//         model->train();
//         for (int epoch = 0; epoch < 10; ++epoch) {
//             float total_loss = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 std::vector<torch::Dict<std::string, torch::Tensor>> targets;
//                 for (const auto& target : batch.target) {
//                     torch::Dict<std::string, torch::Tensor> t;
//                     t.insert("boxes", target.at("boxes").to(device));
//                     t.insert("labels", target.at("labels").to(device));
//                     targets.push_back(t);
//                 }
//
//                 optimizer.zero_grad();
//                 auto [loss, _, _] = model->forward(images, targets);
//                 loss.backward();
//                 optimizer.step();
//
//                 total_loss += loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
//         }
//
//         // Save model
//         torch::save(model, "faster_rcnn_trained.pt");
//         std::cout << "Training complete. Model saved as faster_rcnn_trained.pt" << std::endl;
//
//         // Inference example
//         model->eval();
//         cv::Mat image = cv::imread("test_image.jpg");
//         if (image.empty()) {
//             std::cerr << "Error: Could not load test image." << std::endl;
//             return -1;
//         }
//         cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
//         image.convertTo(image, CV_32F, 1.0 / 255.0);
//         torch::Tensor img_tensor = torch::from_blob(
//             image.data, {1, image.rows, image.cols, 3}, torch::kFloat32
//         ).permute({0, 3, 1, 2}).to(device);
//
//         auto [_, cls_scores, bbox_deltas] = model->forward(img_tensor);
//         cls_scores = torch::softmax(cls_scores, 1);
//         auto max_scores = std::get<1>(torch::max(cls_scores, 1));
//         for (int i = 0; i < max_scores.size(0); ++i) {
//             float score = max_scores[i].item<float>();
//             if (score > 0.5) {
//                 int label = cls_scores[i].argmax().item<int>();
//                 std::cout << "Detected: " << CLASSES[label] << " with confidence " << score << std::endl;
//             }
//         }
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
    struct FastRCNN : xt::Cloneable<FastRCNN>
    {
    private:

    public:
        FastRCNN(int num_classes /* classes */, int in_channels = 3/* input channels */);

        FastRCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape);

        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
        void reset() override;
    };

}