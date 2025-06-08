#include "include/models/computer_vision/object_detection/ssd.h"


using namespace std;
//SSD GROK


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // Lightweight CNN Backbone
// struct BackboneImpl : torch::nn::Module {
//     BackboneImpl(int in_channels) {
//         // Stage 1
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 3).stride(1).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         // Stage 2
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         // Stage 3
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
//         // Extra layers for multi-scale
//         conv4 = register_module("conv4", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 3).stride(2).padding(1)));
//         bn4 = register_module("bn4", torch::nn::BatchNorm2d(256));
//     }
//
//     std::vector<torch::Tensor> forward(torch::Tensor x) {
//         // x: [batch, in_channels, 64, 64]
//         auto f1 = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 64, 64, 64]
//         auto f2 = torch::relu(bn2->forward(conv2->forward(f1))); // [batch, 128, 32, 32]
//         auto f3 = torch::relu(bn3->forward(conv3->forward(f2))); // [batch, 256, 16, 16]
//         auto f4 = torch::relu(bn4->forward(conv4->forward(f3))); // [batch, 256, 8, 8]
//         return {f2, f3, f4}; // Multi-scale feature maps
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr};
// };
// TORCH_MODULE(Backbone);
//
// // Classification Head
// struct ClassificationHeadImpl : torch::nn::Module {
//     ClassificationHeadImpl(int in_channels, int num_anchors, int num_classes)
//         : num_anchors_(num_anchors), num_classes_(num_classes) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, num_anchors * num_classes, 3).padding(1)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, h, w]
//         x = conv->forward(x); // [batch, num_anchors*num_classes, h, w]
//         return x.permute({0, 2, 3, 1}).reshape({-1, num_classes_}); // [batch*h*w*num_anchors, num_classes]
//     }
//
//     int num_anchors_, num_classes_;
//     torch::nn::Conv2d conv{nullptr};
// };
// TORCH_MODULE(ClassificationHead);
//
// // Box Regression Head
// struct BoxRegressionHeadImpl : torch::nn::Module {
//     BoxRegressionHeadImpl(int in_channels, int num_anchors) : num_anchors_(num_anchors) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, num_anchors * 4, 3).padding(1)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, h, w]
//         x = conv->forward(x); // [batch, num_anchors*4, h, w]
//         return x.permute({0, 2, 3, 1}).reshape({-1, 4}); // [batch*h*w*num_anchors, 4]
//     }
//
//     int num_anchors_;
//     torch::nn::Conv2d conv{nullptr};
// };
// TORCH_MODULE(BoxRegressionHead);
//
// // SSD Model
// struct SSDImpl : torch::nn::Module {
//     SSDImpl(int in_channels, int num_classes, int num_anchors = 4)
//         : num_classes_(num_classes), num_anchors_(num_anchors) {
//         backbone = register_module("backbone", Backbone(in_channels));
//         cls_heads = torch::nn::ModuleList({
//             register_module("cls_head1", ClassificationHead(128, num_anchors, num_classes)),
//             register_module("cls_head2", ClassificationHead(256, num_anchors, num_classes)),
//             register_module("cls_head3", ClassificationHead(256, num_anchors, num_classes))
//         });
//         box_heads = torch::nn::ModuleList({
//             register_module("box_head1", BoxRegressionHead(128, num_anchors)),
//             register_module("box_head2", BoxRegressionHead(256, num_anchors)),
//             register_module("box_head3", BoxRegressionHead(256, num_anchors))
//         });
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x,
//                                                     const std::vector<torch::Tensor>& gt_boxes = {},
//                                                     const std::vector<torch::Tensor>& gt_classes = {}) {
//         // x: [batch, in_channels, 64, 64]
//         auto features = backbone->forward(x); // {f2, f3, f4}
//
//         std::vector<torch::Tensor> cls_logits, bbox_deltas;
//         for (size_t i = 0; i < features.size(); ++i) {
//             cls_logits.push_back(cls_heads[i]->forward(features[i])); // [batch*h*w*num_anchors, num_classes]
//             bbox_deltas.push_back(box_heads[i]->forward(features[i])); // [batch*h*w*num_anchors, 4]
//         }
//         auto all_cls_logits = torch::cat(cls_logits, 0); // [total_anchors, num_classes]
//         auto all_bbox_deltas = torch::cat(bbox_deltas, 0); // [total_anchors, 4]
//
//         if (!gt_boxes.empty()) {
//             auto anchors = generate_anchors(features).to(x.device());
//             auto losses = compute_losses(all_cls_logits, all_bbox_deltas, anchors, gt_boxes, gt_classes);
//             return losses;
//         }
//
//         return {all_cls_logits, all_bbox_deltas};
//     }
//
//     std::vector<torch::Tensor> generate_anchors(const std::vector<torch::Tensor>& features) {
//         std::vector<torch::Tensor> all_anchors;
//         std::vector<std::vector<float>> scales = {{16, 32}, {32, 64}, {64, 128}}; // Scales per feature map
//         std::vector<float> ratios = {0.5, 1.0, 2.0};
//         std::vector<float> strides = {2, 4, 8}; // Strides due to downsampling
//
//         for (size_t i = 0; i < features.size(); ++i) {
//             int h = features[i].size(2);
//             int w = features[i].size(3);
//             float stride = strides[i];
//             std::vector<float> anchor_data;
//
//             for (int y = 0; y < h; ++y) {
//                 for (int x = 0; x < w; ++x) {
//                     for (float scale : scales[i]) {
//                         for (float ratio : ratios) {
//                             float w_a = scale * std::sqrt(ratio) / 64.0;
//                             float h_a = scale / std::sqrt(ratio) / 64.0;
//                             float cx = (x + 0.5) * stride / 64.0;
//                             float cy = (y + 0.5) * stride / 64.0;
//                             anchor_data.insert(anchor_data.end(), {cx, cy, w_a, h_a});
//                         }
//                     }
//                 }
//             }
//             all_anchors.push_back(torch::tensor(anchor_data).reshape({-1, 4}));
//         }
//         return all_anchors;
//     }
//
//     std::pair<torch::Tensor, torch::Tensor> compute_losses(const torch::Tensor& cls_logits,
//                                                           const torch::Tensor& bbox_deltas,
//                                                           const std::vector<torch::Tensor>& anchors,
//                                                           const std::vector<torch::Tensor>& gt_boxes,
//                                                           const std::vector<torch::Tensor>& gt_classes) {
//         auto all_anchors = torch::cat(anchors, 0); // [total_anchors, 4]
//         std::vector<int> labels;
//         std::vector<torch::Tensor> target_deltas;
//
//         for (int b = 0; b < gt_boxes.size(); ++b) {
//             auto ious = compute_iou(all_anchors, gt_boxes[b]);
//             auto max_iou = ious.max(1);
//             auto max_indices = max_iou.indices();
//             auto max_values = max_iou.values();
//
//             for (int i = 0; i < all_anchors.size(0); ++i) {
//                 if (max_values[i].item<float>() > 0.5) {
//                     labels.push_back(gt_classes[b][max_indices[i]].item<int>());
//                     auto gt = gt_boxes[b][max_indices[i]];
//                     auto anchor = all_anchors[i];
//                     auto dx = (gt[0] - anchor[0]) / anchor[2];
//                     auto dy = (gt[1] - anchor[1]) / anchor[3];
//                     auto dw = torch::log(gt[2] / anchor[2]);
//                     auto dh = torch::log(gt[3] / anchor[3]);
//                     target_deltas.push_back(torch::tensor({dx.item<float>(), dy.item<float>(),
//                                                           dw.item<float>(), dh.item<float>()}));
//                 } else if (max_values[i].item<float>() < 0.3) {
//                     labels.push_back(0); // Background
//                     target_deltas.push_back(torch::zeros({4}));
//                 } else {
//                     labels.push_back(-1); // Ignore
//                     target_deltas.push_back(torch::zeros({4}));
//                 }
//             }
//         }
//
//         auto label_tensor = torch::tensor(labels).to(cls_logits.device());
//         auto delta_tensor = torch::stack(target_deltas).to(bbox_deltas.device());
//
//         // Classification loss
//         auto valid = label_tensor >= 0;
//         auto cls_loss = valid.sum().item<int>() > 0 ?
//             torch::cross_entropy(cls_logits.index_select(0, valid.nonzero().squeeze(-1)),
//                                  label_tensor.index_select(0, valid.nonzero().squeeze(-1))) :
//             torch::tensor(0.0).to(cls_logits.device());
//
//         // Regression loss
//         auto fg = label_tensor > 0;
//         auto reg_loss = fg.sum().item<int>() > 0 ?
//             torch::smooth_l1_loss(bbox_deltas.index_select(0, fg.nonzero().squeeze(-1)),
//                                   delta_tensor.index_select(0, fg.nonzero().squeeze(-1))) :
//             torch::tensor(0.0).to(bbox_deltas.device());
//
//         return {cls_loss, reg_loss};
//     }
//
//     torch::Tensor compute_iou(const torch::Tensor& boxes1, const torch::Tensor& boxes2) {
//         // boxes1: [N, 4], boxes2: [M, 4] (cx, cy, w, h)
//         auto x1 = boxes1.narrow(1, 0, 1) - boxes1.narrow(1, 2, 1) / 2;
//         auto y1 = boxes1.narrow(1, 1, 1) - boxes1.narrow(1, 3, 1) / 2;
//         auto x2 = boxes1.narrow(1, 0, 1) + boxes1.narrow(1, 2, 1) / 2;
//         auto y2 = boxes1.narrow(1, 1, 1) + boxes1.narrow(1, 3, 1) / 2;
//
//         auto x1g = boxes2.narrow(1, 0, 1) - boxes2.narrow(1, 2, 1) / 2;
//         auto y1g = boxes2.narrow(1, 1, 1) - boxes2.narrow(1, 3, 1) / 2;
//         auto x2g = boxes2.narrow(1, 0, 1) + boxes2.narrow(1, 2, 1) / 2;
//         auto y2g = boxes2.narrow(1, 1, 1) + boxes2.narrow(1, 3, 1) / 2;
//
//         auto xi1 = torch::maximum(x1, x1g.transpose(0, 1));
//         auto yi1 = torch::maximum(y1, y1g.transpose(0, 1));
//         auto xi2 = torch::minimum(x2, x2g.transpose(0, 1));
//         auto yi2 = torch::minimum(y2, y2g.transpose(0, 1));
//
//         auto inter_area = torch::clamp(xi2 - xi1, 0) * torch::clamp(yi2 - yi1, 0);
//         auto area1 = boxes1.narrow(1, 2, 1) * boxes1.narrow(1, 3, 1);
//         auto area2 = boxes2.narrow(1, 2, 1) * boxes2.narrow(1, 3, 1);
//         auto union_area = area1 + area2.transpose(0, 1) - inter_area;
//         return inter_area / (union_area + 1e-6);
//     }
//
//     int num_classes_, num_anchors_;
//     Backbone backbone{nullptr};
//     torch::nn::ModuleList cls_heads{nullptr}, box_heads{nullptr};
// };
// TORCH_MODULE(SSD);
//
// // Detection Dataset
// struct DetectionDataset : torch::data::Dataset<DetectionDataset> {
//     DetectionDataset(const std::string& img_dir, const std::string& annot_dir) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//                 std::string annot_path = annot_dir + "/" + entry.path().filename().string() + ".txt";
//                 annot_paths_.push_back(annot_path);
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         // Load image
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//         }
//         cv::resize(image, image, cv::Size(64, 64));
//         image.convertTo(image, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//         // Load annotations (format: class cx cy w h per line)
//         std::ifstream annot_file(annot_paths_[index % annot_paths_.size()]);
//         if (!annot_file.is_open()) {
//             throw std::runtime_error("Failed to open annotation file: " + annot_paths_[index % annot_paths_.size()]);
//         }
//         std::vector<float> boxes;
//         std::vector<int> classes;
//         int cls;
//         float cx, cy, w, h;
//         while (annot_file >> cls >> cx >> cy >> w >> h) {
//             classes.push_back(cls);
//             boxes.insert(boxes.end(), {cx, cy, w, h});
//         }
//         torch::Tensor class_tensor = torch::tensor(classes).unsqueeze(-1); // [n_objects, 1]
//         torch::Tensor box_tensor = torch::from_blob(boxes.data(), {(int)boxes.size() / 4, 4}, torch::kFloat32); // [n_objects, 4]
//
//         return {img_tensor, torch::stack({class_tensor, box_tensor})};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_, annot_paths_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int in_channels = 1;
//         const int num_classes = 2; // Binary detection (1 + background)
//         const int num_anchors = 4;
//         const int batch_size = 4;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         SSD model(in_channels, num_classes, num_anchors);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Load dataset
//         auto dataset = DetectionDataset("./data/images", "./data/annotations")
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         model->train();
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float loss_avg = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 std::vector<torch::Tensor> gt_classes, gt_boxes;
//                 for (int b = 0; b < batch_size; ++b) {
//                     gt_classes.push_back(batch.target[0][b].squeeze(-1)); // [n_objects]
//                     gt_boxes.push_back(batch.target[1][b]); // [n_objects, 4]
//                 }
//
//                 optimizer.zero_grad();
//                 auto [cls_loss, reg_loss] = model->forward(images, gt_boxes, gt_classes);
//                 auto loss = cls_loss + reg_loss;
//                 loss.backward();
//                 optimizer.step();
//
//                 loss_avg += loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Loss: " << loss_avg / batch_count << std::endl;
//
//             // Save model every 10 epochs
//             if ((epoch + 1) % 10 == 0) {
//                 torch::save(model, "ssd_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "ssd.pt");
//         std::cout << "Model saved as ssd.pt" << std::endl;
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
    SSD::SSD(int num_classes, int in_channels)
    {
    }

    SSD::SSD(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void SSD::reset()
    {
    }

    auto SSD::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }
}
