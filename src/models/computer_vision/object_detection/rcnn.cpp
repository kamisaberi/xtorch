#include <models/computer_vision/object_detection/rcnn.h>


using namespace std;

//RCNN GROK


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <random>
//
// // Lightweight CNN Backbone
// struct BackboneImpl : torch::nn::Module {
//     BackboneImpl(int in_channels) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 3).stride(2).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
//         fc = register_module("fc", torch::nn::Linear(256 * 8 * 8, 4096));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, 64, 64]
//         x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 64, 32, 32]
//         x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 128, 16, 16]
//         x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 256, 8, 8]
//         x = x.view({x.size(0), -1}); // [batch, 256*8*8]
//         x = torch::relu(fc->forward(x)); // [batch, 4096]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
//     torch::nn::Linear fc{nullptr};
// };
// TORCH_MODULE(Backbone);
//
// // Classification and Regression Head
// struct RCNNHeadImpl : torch::nn::Module {
//     RCNNHeadImpl(int in_features, int num_classes) {
//         cls_score = register_module("cls_score", torch::nn::Linear(in_features, num_classes + 1)); // +1 for background
//         bbox_pred = register_module("bbox_pred", torch::nn::Linear(in_features, num_classes * 4)); // Box deltas per class
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // x: [num_proposals, in_features]
//         auto cls_logits = cls_score->forward(x); // [num_proposals, num_classes+1]
//         auto bbox_deltas = bbox_pred->forward(x); // [num_proposals, num_classes*4]
//         return {cls_logits, bbox_deltas};
//     }
//
//     torch::nn::Linear cls_score{nullptr}, bbox_pred{nullptr};
// };
// TORCH_MODULE(RCNNHead);
//
// // R-CNN Model
// struct RCNNImpl : torch::nn::Module {
//     RCNNImpl(int in_channels, int num_classes) : num_classes_(num_classes) {
//         backbone = register_module("backbone", Backbone(in_channels));
//         rcnn_head = register_module("rcnn_head", RCNNHead(4096, num_classes));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, const std::vector<torch::Tensor>& proposals = {},
//                                                     const std::vector<torch::Tensor>& gt_boxes = {},
//                                                     const std::vector<torch::Tensor>& gt_classes = {}) {
//         // x: [batch, in_channels, 64, 64]
//         // proposals: [batch][num_proposals, 4] (x1, y1, x2, y2)
//
//         std::vector<torch::Tensor> cls_logits, bbox_deltas;
//         for (int b = 0; b < x.size(0); ++b) {
//             auto img = x[b].unsqueeze(0); // [1, in_channels, 64, 64]
//             std::vector<torch::Tensor> roi_features;
//
//             // Extract ROIs
//             auto b_proposals = proposals.empty() ? generate_proposals(img) : proposals[b];
//             for (int i = 0; i < b_proposals.size(0); ++i) {
//                 float x1 = b_proposals[i][0].item<float>() * 64;
//                 float y1 = b_proposals[i][1].item<float>() * 64;
//                 float x2 = b_proposals[i][2].item<float>() * 64;
//                 float y2 = b_proposals[i][3].item<float>() * 64;
//
//                 x1 = std::max(0.0f, x1);
//                 y1 = std::max(0.0f, y1);
//                 x2 = std::min(64.0f, x2);
//                 y2 = std::min(64.0f, y2);
//
//                 if (x2 <= x1 || y2 <= y1) continue;
//
//                 // Crop and resize ROI
//                 auto roi = img.slice(2, static_cast<int>(y1), static_cast<int>(y2))
//                              .slice(3, static_cast<int>(x1), static_cast<int>(x2)); // [1, in_channels, h, w]
//                 roi = torch::adaptive_avg_pool2d(roi, {64, 64}); // [1, in_channels, 64, 64]
//
//                 // Extract features
//                 auto feat = backbone->forward(roi); // [1, 4096]
//                 roi_features.push_back(feat);
//             }
//
//             if (roi_features.empty()) continue;
//
//             auto batch_features = torch::cat(roi_features, 0); // [num_proposals, 4096]
//             auto [b_cls_logits, b_bbox_deltas] = rcnn_head->forward(batch_features);
//             cls_logits.push_back(b_cls_logits);
//             bbox_deltas.push_back(b_bbox_deltas);
//         }
//
//         if (cls_logits.empty()) {
//             return {torch::tensor(0.0).to(x.device()), torch::tensor(0.0).to(x.device())};
//         }
//
//         auto all_cls_logits = torch::cat(cls_logits, 0); // [total_proposals, num_classes+1]
//         auto all_bbox_deltas = torch::cat(bbox_deltas, 0); // [total_proposals, num_classes*4]
//
//         if (!gt_boxes.empty()) {
//             auto losses = compute_losses(all_cls_logits, all_bbox_deltas, proposals, gt_boxes, gt_classes);
//             return {losses.first, losses.second};
//         }
//
//         return {all_cls_logits, all_bbox_deltas};
//     }
//
//     torch::Tensor generate_proposals(const torch::Tensor& img) {
//         // Simplified selective search: generate random proposals
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<float> dist(0.0, 1.0);
//
//         std::vector<float> proposals;
//         for (int i = 0; i < 20; ++i) { // Generate 20 proposals
//             float x1 = dist(gen) * 0.8;
//             float y1 = dist(gen) * 0.8;
//             float w = dist(gen) * (1.0 - x1) * 0.5 + 0.1;
//             float h = dist(gen) * (1.0 - y1) * 0.5 + 0.1;
//             float x2 = x1 + w;
//             float y2 = y1 + h;
//             proposals.insert(proposals.end(), {x1, y1, x2, y2});
//         }
//         return torch::tensor(proposals).reshape({-1, 4}); // [num_proposals, 4]
//     }
//
//     std::pair<torch::Tensor, torch::Tensor> compute_losses(const torch::Tensor& cls_logits, const torch::Tensor& bbox_deltas,
//                                                           const std::vector<torch::Tensor>& proposals,
//                                                           const std::vector<torch::Tensor>& gt_boxes,
//                                                           const std::vector<torch::Tensor>& gt_classes) {
//         std::vector<int> labels;
//         std::vector<torch::Tensor> target_deltas;
//         int proposal_idx = 0;
//
//         for (int b = 0; b < gt_boxes.size(); ++b) {
//             auto b_proposals = proposals[b];
//             auto b_gt_boxes = gt_boxes[b];
//             auto ious = compute_iou(b_proposals, b_gt_boxes);
//             auto max_iou = ious.max(1);
//             auto max_indices = max_iou.indices();
//             auto max_values = max_iou.values();
//
//             for (int i = 0; i < b_proposals.size(0); ++i) {
//                 if (max_values[i].item<float>() > 0.5) {
//                     int cls = gt_classes[b][max_indices[i]].item<int>();
//                     labels.push_back(cls);
//                     auto gt = b_gt_boxes[max_indices[i]];
//                     auto prop = b_proposals[i];
//                     auto cx_gt = (gt[0] + gt[2]) / 2;
//                     auto cy_gt = (gt[1] + gt[3]) / 2;
//                     auto cx_prop = (prop[0] + prop[2]) / 2;
//                     auto cy_prop = (prop[1] + prop[3]) / 2;
//                     auto w_gt = gt[2] - gt[0];
//                     auto h_gt = gt[3] - gt[1];
//                     auto w_prop = prop[2] - prop[0];
//                     auto h_prop = prop[3] - prop[1];
//                     auto dx = (cx_gt - cx_prop) / w_prop;
//                     auto dy = (cy_gt - cy_prop) / h_prop;
//                     auto dw = torch::log(w_gt / w_prop);
//                     auto dh = torch::log(h_gt / h_prop);
//                     target_deltas.push_back(torch::tensor({dx.item<float>(), dy.item<float>(),
//                                                           dw.item<float>(), dh.item<float>()}));
//                 } else {
//                     labels.push_back(0); // Background
//                     target_deltas.push_back(torch::zeros({4}));
//                 }
//                 proposal_idx++;
//             }
//         }
//
//         if (labels.empty()) {
//             return {torch::tensor(0.0).to(cls_logits.device()), torch::tensor(0.0).to(bbox_deltas.device())};
//         }
//
//         auto label_tensor = torch::tensor(labels).to(cls_logits.device());
//         auto delta_tensor = torch::stack(target_deltas).to(bbox_deltas.device());
//
//         // Classification loss
//         auto cls_loss = torch::cross_entropy(cls_logits, label_tensor);
//
//         // Regression loss (only for foreground)
//         auto fg = label_tensor > 0;
//         auto reg_loss = fg.sum().item<int>() > 0 ?
//             torch::smooth_l1_loss(bbox_deltas.index_select(0, fg.nonzero().squeeze(-1))
//                                      .reshape({-1, num_classes_, 4})
//                                      .index_select(1, label_tensor.index_select(0, fg.nonzero().squeeze(-1)) - 1),
//                                   delta_tensor.index_select(0, fg.nonzero().squeeze(-1))) :
//             torch::tensor(0.0).to(bbox_deltas.device());
//
//         return {cls_loss, reg_loss};
//     }
//
//     torch::Tensor compute_iou(const torch::Tensor& boxes1, const torch::Tensor& boxes2) {
//         // boxes1: [N, 4] (x1, y1, x2, y2), boxes2: [M, 4]
//         auto x1 = torch::maximum(boxes1.narrow(1, 0, 1), boxes2.narrow(1, 0, 1).transpose(0, 1));
//         auto y1 = torch::maximum(boxes1.narrow(1, 1, 1), boxes2.narrow(1, 1, 1).transpose(0, 1));
//         auto x2 = torch::minimum(boxes1.narrow(1, 2, 1), boxes2.narrow(1, 2, 1).transpose(0, 1));
//         auto y2 = torch::minimum(boxes1.narrow(1, 3, 1), boxes2.narrow(1, 3, 1).transpose(0, 1));
//
//         auto inter_area = torch::clamp(x2 - x1, 0) * torch::clamp(y2 - y1, 0);
//         auto area1 = (boxes1.narrow(1, 2, 1) - boxes1.narrow(1, 0, 1)) *
//                      (boxes1.narrow(1, 3, 1) - boxes1.narrow(1, 1, 1));
//         auto area2 = (boxes2.narrow(1, 2, 1) - boxes2.narrow(1, 0, 1)) *
//                      boxes2.narrow(1, 3, 1) - boxes2.narrow(1, 1, 1));
//         auto union_area = area1[0] + area2[0].transpose(0, 1) - inter_area;
//         return inter_area / (union_area + 1e-6);
//     }
//
//     int num_classes_;
//     Backbone backbone{nullptr};
//     RCNNHead rcnn_head{nullptr};
// };
// TORCH_MODULE(RCNN);
//
// // Detection Dataset
// struct DetectionDataset : torch::data::Dataset<DetectionDataset> {
//     DetectionDataset(const std::string& img_dir, const std::string& annot_dir) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().path() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//                 std::string annot_path = annot_dir + "/" + entry.path().filename().string() + ".txt";
//                 annot_paths_.push_back(annot_path);
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) const override {
//         // Load image
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//             }
//         }
//         cv::resize(image, image, cv::Size(64, 64)); // Resize to 64x64
//         image.convertTo(image, CV_ConvertTo32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//         // Load annotations (format: class cx cy w h)
//         std::ifstream annot_file(annot_paths_[index % annot_paths_.size()]);
//         if (!annot_file.is_open()) {
//             throw std::runtime_error("Failed to open annotation file: " + annot_paths_[index % annot_paths_.size()]);
//         }
//         std::vector<float> boxes;
//         std::vector<int> classes;
//         int cls;
//         float cx, cx, cy, w, h;
//         while (annot_file >> cls >> cx >> cy >> w >> h) {
//             classes.push_back(cls);
//             boxes.insert(boxes.end(), {cx - w/2, cy - h/2, cx + w/2, cy + h/2}); // Convert to x1, y1, x2, y2)
//         }
//         torch::Tensor class_tensor = torch::tensor(classes).unsqueeze(-1); // [n_objects, 1]
//         torch::Tensor box_tensor = torch::from_blob(boxes.data(), {(int)boxes.size() / 4, 4}, torch::kFloat32); // [batch_size, 4]
//
//         return {img_tensor, torch::stack({class_tensor, box_tensor})};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> paths_, image_paths_, annot_paths_;
// };
// vector<string> images;
//
// // Main function
// int main() {
//     try {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int in_channels = 1;
//         const int num_classes = 1; // Binary detection
//         const int batch_size = 4;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         RCNN model(in_channels, num_classes);
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
//                 std::vector<torch::Tensor> gt_boxes, gt_classes;
//                 for (int b = 0; b < batch_size; ++b) {
//                     gt_classes.push_back(batch.target[0][b].squeeze(-1)); // [n_objects]
//                     gt_boxes.push_back(batch.target[1][b]); // [n_objects, 4]
//                 }
//
//                 // Generate proposals
//                 std::vector<torch::Tensor> proposals;
//                 for (int b = 0; b < batch_size; ++b) {
//                     proposals.push_back(model->generate_proposals(images[b].unsqueeze(0)).to(device));
//                 }
//
//                 optimizer.zero_grad();
//                 auto [cls_loss, reg_loss] = model->forward(images, proposals, gt_boxes, gt_classes);
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
//                 torch::save(model, "rcnn_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "rcnn.pt");
//         std::cout << "Model saved as rcnn.pt" << std::endl;
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
    RCNN::RCNN(int num_classes, int in_channels)
    {
    }

    RCNN::RCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void RCNN::reset()
    {
    }

    auto RCNN::forward(std::initializer_list<std::any> tensors) -> std::any
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
