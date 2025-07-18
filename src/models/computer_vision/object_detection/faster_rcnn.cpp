#include <models/computer_vision/object_detection/faster_rcnn.h>


using namespace std;

//FasterRCNN GROK

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
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 3).stride(2).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, 64, 64]
//         x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 64, 32, 32]
//         x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 128, 16, 16]
//         x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 256, 8, 8]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
// };
// TORCH_MODULE(Backbone);
//
// // Region Proposal Network (RPN)
// struct RPNImpl : torch::nn::Module {
//     RPNImpl(int in_channels, int mid_channels, int num_anchors) : num_anchors_(num_anchors) {
//         conv = register_module("conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, mid_channels, 3).padding(1)));
//         cls_score = register_module("cls_score", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(mid_channels, num_anchors * 2, 1))); // Objectness: foreground/background
//         bbox_pred = register_module("bbox_pred", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(mid_channels, num_anchors * 4, 1))); // Box deltas
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // x: [batch, in_channels, h, w]
//         x = torch::relu(conv->forward(x)); // [batch, mid_channels, h, w]
//         auto cls_logits = cls_score->forward(x); // [batch, num_anchors*2, h, w]
//         auto bbox_deltas = bbox_pred->forward(x); // [batch, num_anchors*4, h, w]
//
//         // Reshape for processing
//         cls_logits = cls_logits.permute({0, 2, 3, 1}).reshape({-1, 2}); // [batch*h*w*num_anchors, 2]
//         bbox_deltas = bbox_deltas.permute({0, 2, 3, 1}).reshape({-1, 4}); // [batch*h*w*num_anchors, 4]
//         return {cls_logits, bbox_deltas};
//     }
//
//     int num_anchors_;
//     torch::nn::Conv2d conv{nullptr}, cls_score{nullptr}, bbox_pred{nullptr};
// };
// TORCH_MODULE(RPN);
//
// // ROI Pooling (Simplified)
// struct RoiPoolImpl : torch::nn::Module {
//     RoiPoolImpl(int output_size) : output_size_(output_size) {}
//
//     torch::Tensor forward(const torch::Tensor& features, const torch::Tensor& rois) {
//         // features: [batch, channels, h, w], rois: [num_rois, 5] (batch_idx, x1, y1, x2, y2)
//         std::vector<torch::Tensor> pooled;
//         int batch_size = features.size(0);
//         int channels = features.size(1);
//         int feat_h = features.size(2);
//         int feat_w = features.size(3);
//
//         for (int i = 0; i < rois.size(0); ++i) {
//             int batch_idx = rois[i][0].item<int>();
//             float x1 = rois[i][1].item<float>() * feat_w;
//             float y1 = rois[i][2].item<float>() * feat_h;
//             float x2 = rois[i][3].item<float>() * feat_w;
//             float y2 = rois[i][4].item<float>() * feat_h;
//
//             int roi_x1 = std::max(0, static_cast<int>(std::floor(x1)));
//             int roi_y1 = std::max(0, static_cast<int>(std::floor(y1)));
//             int roi_x2 = std::min(feat_w, static_cast<int>(std::ceil(x2)));
//             int roi_y2 = std::min(feat_h, static_cast<int>(std::ceil(y2)));
//
//             if (roi_x2 <= roi_x1 || roi_y2 <= roi_y1) {
//                 pooled.push_back(torch::zeros({channels, output_size_, output_size_}, features.options()));
//                 continue;
//             }
//
//             auto roi = features[batch_idx].slice(1, roi_y1, roi_y2).slice(2, roi_x1, roi_x2); // [channels, roi_h, roi_w]
//             auto pooled_roi = torch::adaptive_max_pool2d(roi, {output_size_, output_size_}); // [channels, output_size, output_size]
//             pooled.push_back(pooled_roi);
//         }
//
//         return torch::stack(pooled); // [num_rois, channels, output_size, output_size]
//     }
//
//     int output_size_;
// };
// TORCH_MODULE(RoiPool);
//
// // Fast R-CNN Head
// struct FastRCNNHeadImpl : torch::nn::Module {
//     FastRCNNHeadImpl(int in_channels, int num_classes) {
//         fc1 = register_module("fc1", torch::nn::Linear(in_channels, 1024));
//         fc2 = register_module("fc2", torch::nn::Linear(1024, 1024));
//         cls_score = register_module("cls_score", torch::nn::Linear(1024, num_classes + 1)); // +1 for background
//         bbox_pred = register_module("bbox_pred", torch::nn::Linear(1024, (num_classes + 1) * 4));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // x: [num_rois, channels, output_size, output_size]
//         x = x.flatten(1); // [num_rois, channels*output_size*output_size]
//         x = torch::relu(fc1->forward(x)); // [num_rois, 1024]
//         x = torch::relu(fc2->forward(x)); // [num_rois, 1024]
//         auto cls_logits = cls_score->forward(x); // [num_rois, num_classes+1]
//         auto bbox_deltas = bbox_pred->forward(x); // [num_rois, (num_classes+1)*4]
//         return {cls_logits, bbox_deltas};
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr}, cls_score{nullptr}, bbox_pred{nullptr};
// };
// TORCH_MODULE(FastRCNNHead);
//
// // Faster R-CNN Model
// struct FasterRCNNImpl : torch::nn::Module {
//     FasterRCNNImpl(int in_channels, int num_classes, int num_anchors = 3) : num_classes_(num_classes), num_anchors_(num_anchors) {
//         backbone = register_module("backbone", Backbone(in_channels));
//         rpn = register_module("rpn", RPN(256, 256, num_anchors));
//         roi_pool = register_module("roi_pool", RoiPool(7));
//         fast_rcnn_head = register_module("fast_rcnn_head", FastRCNNHead(256 * 7 * 7, num_classes));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x, const std::vector<torch::Tensor>& gt_boxes = {}, const std::vector<torch::Tensor>& gt_classes = {}) {
//         // x: [batch, in_channels, 64, 64]
//         auto features = backbone->forward(x); // [batch, 256, 8, 8]
//
//         // RPN
//         auto [rpn_cls_logits, rpn_bbox_deltas] = rpn->forward(features); // [batch*h*w*num_anchors, 2], [batch*h*w*num_anchors, 4]
//
//         // Generate anchors
//         auto anchors = generate_anchors(features.size(2), features.size(3)).to(x.device());
//
//         // Proposal generation (simplified, select top proposals)
//         auto proposals = generate_proposals(anchors, rpn_cls_logits, rpn_bbox_deltas, x.size(0));
//
//         // ROI Pooling
//         auto roi_features = roi_pool->forward(features, proposals); // [num_proposals, 256, 7, 7]
//
//         // Fast R-CNN Head
//         auto [cls_logits, bbox_deltas] = fast_rcnn_head->forward(roi_features); // [num_proposals, num_classes+1], [num_proposals, (num_classes+1)*4]
//
//         if (!gt_boxes.empty()) {
//             // Compute losses
//             auto rpn_loss = compute_rpn_loss(rpn_cls_logits, rpn_bbox_deltas, anchors, gt_boxes);
//             auto fast_rcnn_loss = compute_fast_rcnn_loss(cls_logits, bbox_deltas, proposals, gt_boxes, gt_classes);
//             return {rpn_loss.first, rpn_loss.second, fast_rcnn_loss.first, fast_rcnn_loss.second};
//         }
//
//         return {cls_logits, bbox_deltas, torch::Tensor(), torch::Tensor()};
//     }
//
//     torch::Tensor generate_anchors(int height, int width) {
//         std::vector<float> anchor_data;
//         std::vector<float> scales = {8, 16, 32};
//         std::vector<float> ratios = {0.5, 1.0, 2.0};
//         float stride = 8.0; // Due to backbone downsampling (64/8)
//
//         for (int y = 0; y < height; ++y) {
//             for (int x = 0; x < width; ++x) {
//                 for (float scale : scales) {
//                     for (float ratio : ratios) {
//                         float w = scale * std::sqrt(ratio);
//                         float h = scale / std::sqrt(ratio);
//                         float cx = (x + 0.5) * stride / 64.0;
//                         float cy = (y + 0.5) * stride / 64.0;
//                         anchor_data.insert(anchor_data.end(), {cx, cy, w / 64.0, h / 64.0});
//                     }
//                 }
//             }
//         }
//         return torch::tensor(anchor_data).reshape({-1, 4}); // [h*w*num_anchors, 4]
//     }
//
//     torch::Tensor generate_proposals(const torch::Tensor& anchors, const torch::Tensor& cls_logits, const torch::Tensor& bbox_deltas, int batch_size) {
//         // Simplified proposal generation
//         auto scores = torch::softmax(cls_logits, 1).narrow(1, 1, 1).squeeze(-1); // [batch*h*w*num_anchors]
//         auto deltas = bbox_deltas; // [batch*h*w*num_anchors, 4]
//         std::vector<torch::Tensor> batch_proposals;
//
//         int num_anchors = anchors.size(0) / batch_size;
//         for (int b = 0; b < batch_size; ++b) {
//             auto b_scores = scores.slice(0, b * num_anchors, (b + 1) * num_anchors);
//             auto b_deltas = deltas.slice(0, b * num_anchors, (b + 1) * num_anchors);
//             auto b_anchors = anchors.slice(0, b * num_anchors, (b + 1) * num_anchors);
//
//             // Apply deltas
//             auto boxes = apply_deltas(b_anchors, b_deltas);
//
//             // Clip boxes to [0, 1]
//             boxes = torch::clamp(boxes, 0, 1);
//
//             // Select top 10 proposals (simplified NMS)
//             auto topk = std::min(10, static_cast<int>(b_scores.size(0)));
//             auto top_scores = b_scores.topk(topk, 0);
//             auto top_boxes = boxes.index_select(0, top_scores.indices());
//
//             // Create ROIs: [batch_idx, x1, y1, x2, y2]
//             auto batch_idx = torch::full({topk, 1}, b, boxes.options());
//             auto x1 = top_boxes.narrow(1, 0, 1) - top_boxes.narrow(1, 2, 1) / 2;
//             auto y1 = top_boxes.narrow(1, 1, 1) - top_boxes.narrow(1, 3, 1) / 2;
//             auto x2 = top_boxes.narrow(1, 0, 1) + top_boxes.narrow(1, 2, 1) / 2;
//             auto y2 = top_boxes.narrow(1, 1, 1) + top_boxes.narrow(1, 3, 1) / 2;
//             auto rois = torch::cat({batch_idx, x1, y1, x2, y2}, 1);
//             batch_proposals.push_back(rois);
//         }
//
//         return torch::cat(batch_proposals, 0); // [num_proposals, 5]
//     }
//
//     torch::Tensor apply_deltas(const torch::Tensor& anchors, const torch::Tensor& deltas) {
//         // anchors, deltas: [N, 4] (cx, cy, w, h)
//         auto cx = anchors.narrow(1, 0, 1) + deltas.narrow(1, 0, 1) * anchors.narrow(1, 2, 1);
//         auto cy = anchors.narrow(1, 1, 1) + deltas.narrow(1, 1, 1) * anchors.narrow(1, 3, 1);
//         auto w = anchors.narrow(1, 2, 1) * torch::exp(deltas.narrow(1, 2, 1));
//         auto h = anchors.narrow(1, 3, 1) * torch::exp(deltas.narrow(1, 3, 1));
//         return torch::cat({cx, cy, w, h}, 1);
//     }
//
//     std::pair<torch::Tensor, torch::Tensor> compute_rpn_loss(const torch::Tensor& cls_logits, const torch::Tensor& bbox_deltas,
//                                                             const torch::Tensor& anchors, const std::vector<torch::Tensor>& gt_boxes) {
//         // Assign anchors to ground-truth
//         std::vector<int> labels;
//         std::vector<torch::Tensor> target_deltas;
//         int num_anchors = anchors.size(0) / cls_logits.size(0) * 2; // Per batch
//         for (int b = 0; b < gt_boxes.size(); ++b) {
//             auto b_anchors = anchors.slice(0, b * num_anchors / 2, (b + 1) * num_anchors / 2);
//             auto ious = compute_iou(b_anchors, gt_boxes[b]);
//             auto max_iou = ious.max(1);
//             auto max_indices = max_iou.indices();
//             auto max_values = max_iou.values();
//
//             for (int i = 0; i < b_anchors.size(0); ++i) {
//                 if (max_values[i].item<float>() > 0.7) {
//                     labels.push_back(1); // Foreground
//                     auto gt = gt_boxes[b][max_indices[i]];
//                     auto anchor = b_anchors[i];
//                     auto dx = (gt[0] - anchor[0]) / anchor[2];
//                     auto dy = (gt[1] - anchor[1]) / anchor[3];
//                     auto dw = torch::log(gt[2] / anchor[2]);
//                     auto dh = torch::log(gt[3] / anchor[3]);
//                     target_deltas.push_back(torch::tensor({dx.item<float>(), dy.item<float>(), dw.item<float>(), dh.item<float>()}));
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
//         // RPN classification loss
//         auto valid = label_tensor >= 0;
//         auto cls_loss = torch::cross_entropy(cls_logits.index_select(0, valid.nonzero().squeeze(-1)),
//                                             label_tensor.index_select(0, valid.nonzero().squeeze(-1)));
//
//         // RPN regression loss
//         auto fg = label_tensor == 1;
//         auto reg_loss = torch::smooth_l1_loss(bbox_deltas.index_select(0, fg.nonzero().squeeze(-1)),
//                                              delta_tensor.index_select(0, fg.nonzero().squeeze(-1)));
//
//         return {cls_loss, reg_loss};
//     }
//
//     std::pair<torch::Tensor, torch::Tensor> compute_fast_rcnn_loss(const torch::Tensor& cls_logits, const torch::Tensor& bbox_deltas,
//                                                                   const torch::Tensor& proposals, const std::vector<torch::Tensor>& gt_boxes,
//                                                                   const std::vector<torch::Tensor>& gt_classes) {
//         // Match proposals to ground-truth
//         std::vector<int> labels;
//         std::vector<torch::Tensor> target_deltas;
//         int proposal_idx = 0;
//
//         for (int b = 0; b < gt_boxes.size(); ++b) {
//             auto b_proposals = proposals.index_select(0, (proposals.narrow(1, 0, 1) == b).nonzero().squeeze(-1));
//             if (b_proposals.size(0) == 0) continue;
//
//             auto boxes = torch::cat({
//                 b_proposals.narrow(1, 1, 1) - b_proposals.narrow(1, 3, 1) / 2, // cx
//                 b_proposals.narrow(1, 2, 1) - b_proposals.narrow(1, 4, 1) / 2, // cy
//                 b_proposals.narrow(1, 3, 1) - b_proposals.narrow(1, 1, 1),     // w
//                 b_proposals.narrow(1, 4, 1) - b_proposals.narrow(1, 2, 1)      // h
//             }, 1);
//
//             auto ious = compute_iou(boxes, gt_boxes[b]);
//             auto max_iou = ious.max(1);
//             auto max_indices = max_iou.indices();
//             auto max_values = max_iou.values();
//
//             for (int i = 0; i < boxes.size(0); ++i) {
//                 if (max_values[i].item<float>() > 0.5) {
//                     labels.push_back(gt_classes[b][max_indices[i]].item<int>());
//                     auto gt = gt_boxes[b][max_indices[i]];
//                     auto prop = boxes[i];
//                     auto dx = (gt[0] - prop[0]) / prop[2];
//                     auto dy = (gt[1] - prop[1]) / prop[3];
//                     auto dw = torch::log(gt[2] / prop[2]);
//                     auto dh = torch::log(gt[3] / prop[3]);
//                     target_deltas.push_back(torch::tensor({dx.item<float>(), dy.item<float>, dw.item<float>(), dh.item<float>()}));
//                 } else {
//                     labels.push_back(0); // Background
//                     target_deltas.push_back(torch::zeros({4}));
//                 }
//                 proposal_idx++;
//             }
//         }
//
//         if (labels.empty()) return {torch::tensor(0.0).to(cls_logits.device()), torch::tensor(0.0).to(bbox_deltas.device())};
//
//         auto label_tensor = torch::tensor(labels).to(cls_logits.device());
//         auto delta_tensor = torch::stack(target_deltas).to(bbox_deltas.device());
//
//         // Fast R-CNN classification loss
//         auto cls_loss = torch::cross_entropy(cls_logits, label_tensor);
//
//         // Fast R-CNN regression loss
//         auto fg = label_tensor > 0;
//         auto reg_loss = fg.sum().item<int>() > 0 ?
//             torch::smooth_l1_loss(bbox_deltas.index_select(0, fg.nonzero().squeeze(-1)).reshape({-1, num_classes_ + 1, 4})
//                                   .index_select(1, label_tensor.index_select(0, fg.nonzero().squeeze(-1))),
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
//     RPN rpn{nullptr};
//     RoiPool roi_pool{nullptr};
//     FastRCNNHead fast_rcnn_head{nullptr};
// };
// TORCH_MODULE(FasterRCNN);
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
//         cv::resize(image, image, cv::Size(64, 64)); // Resize to 64x64
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
//         const int num_classes = 1; // Binary detection (e.g., digit vs. background)
//         const int num_anchors = 3;
//         const int batch_size = 4;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         FasterRCNN model(in_channels, num_classes, num_anchors);
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
//                 auto [rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss] = model->forward(images, gt_boxes, gt_classes);
//
//                 auto loss = rpn_cls_loss + rpn_reg_loss + fast_rcnn_cls_loss + fast_rcnn_reg_loss;
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
//                 torch::save(model, "fasterrcnn_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "fasterrcnn.pt");
//         std::cout << "Model saved as fasterrcnn.pt" << std::endl;
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
    FasterRCNN::FasterRCNN(int num_classes, int in_channels)
    {
    }

    FasterRCNN::FasterRCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void FasterRCNN::reset()
    {
    }

    auto FasterRCNN::forward(std::initializer_list<std::any> tensors) -> std::any
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
