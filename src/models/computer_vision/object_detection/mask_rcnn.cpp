#include "include/models/computer_vision/object_detection/mask_rcnn.h"


using namespace std;

//MaskRCNN GROK

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
//             torch::nn::Conv2dOptions(mid_channels, num_anchors * 2, 1))); // Objectness
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
// // ROI Align (Simplified)
// struct RoiAlignImpl : torch::nn::Module {
//     RoiAlignImpl(int output_size, float spatial_scale)
//         : output_size_(output_size), spatial_scale_(spatial_scale) {}
//
//     torch::Tensor forward(const torch::Tensor& features, const torch::Tensor& rois) {
//         // features: [batch, channels, h, w], rois: [num_rois, 5] (batch_idx, x1, y1, x2, y2)
//         std::vector<torch::Tensor> aligned;
//         int channels = features.size(1);
//
//         for (int i = 0; i < rois.size(0); ++i) {
//             int batch_idx = rois[i][0].item<int>();
//             float x1 = rois[i][1].item<float>() * spatial_scale_;
//             float y1 = rois[i][2].item<float>() * spatial_scale_;
//             float x2 = rois[i][3].item<float>() * spatial_scale_;
//             float y2 = rois[i][4].item<float>() * spatial_scale_;
//
//             // Compute grid for bilinear interpolation
//             float roi_w = x2 - x1;
//             float roi_h = y2 - y1;
//             float bin_size_h = roi_h / (output_size_ - 1);
//             float bin_size_w = roi_w / (output_size_ - 1);
//
//             torch::Tensor roi_out = torch::zeros({channels, output_size_, output_size_}, features.options());
//             for (int oh = 0; oh < output_size_; ++oh) {
//                 for (int ow = 0; ow < output_size_; ++ow) {
//                     float h = y1 + oh * bin_size_h;
//                     float w = x1 + ow * bin_size_w;
//
//                     int h0 = static_cast<int>(std::floor(h));
//                     int w0 = static_cast<int>(std::floor(w));
//                     int h1 = std::min(h0 + 1, static_cast<int>(features.size(2)) - 1);
//                     int w1 = std::min(w0 + 1, static_cast<int>(features.size(3)) - 1);
//
//                     float dh = h - h0;
//                     float dw = w - w0;
//
//                     auto feat = features[batch_idx];
//                     auto v00 = feat.slice(1, h0, h0 + 1).slice(2, w0, w0 + 1);
//                     auto v01 = feat.slice(1, h0, h0 + 1).slice(2, w1, w1 + 1);
//                     auto v10 = feat.slice(1, h1, h1 + 1).slice(2, w0, w0 + 1);
//                     auto v11 = feat.slice(1, h1, h1 + 1).slice(2, w1, w1 + 1);
//
//                     auto interp = v00 * (1 - dh) * (1 - dw) + v01 * (1 - dh) * dw +
//                                   v10 * dh * (1 - dw) + v11 * dh * dw;
//                     roi_out.slice(1, oh, oh + 1).slice(2, ow, ow + 1) = interp;
//                 }
//             }
//             aligned.push_back(roi_out);
//         }
//
//         return torch::stack(aligned); // [num_rois, channels, output_size, output_size]
//     }
//
//     private:
//         int output_size_;
//         float spatial_scale_;
// };
// TORCH_MODULE(RoiAlign);
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
// // Mask Head
// struct MaskHeadImpl : torch::nn::Module {
//     MaskHeadImpl(int in_channels, int num_classes) : num_classes_(num_classes) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 128, 3).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
//         deconv = register_module("deconv", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(128, 128, 2).stride(2)));
//         mask_pred = register_module("mask_pred", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, num_classes, 1)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [num_rois, channels, output_size, output_size]
//         x = torch::relu(conv1->forward(x)); // [num_rois, 128, 7, 7]
//         x = torch::relu(conv2->forward(x)); // [num_rois, 128, 7, 7]
//         x = torch::relu(deconv->forward(x)); // [num_rois, 128, 14, 14]
//         x = mask_pred->forward(x); // [num_rois, num_classes, 14, 14]
//         return x;
//     }
//
//     int num_classes_;
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, mask_pred{nullptr};
//     torch::nn::ConvTranspose2d deconv{nullptr};
// };
// TORCH_MODULE(MaskHead);
//
// // Mask R-CNN Model
// struct MaskRCNNImpl : torch::nn::Module {
//     MaskRCNNImpl(int in_channels, int num_classes, int num_anchors = 3)
//         : num_classes_(num_classes), num_anchors_(num_anchors) {
//         backbone = register_module("backbone", Backbone(in_channels));
//         rpn = register_module("rpn", RPN(256, 256, num_anchors));
//         roi_align = register_module("roi_align", RoiAlign(7, 0.125)); // 1/8 scale due to backbone
//         fast_rcnn_head = register_module("fast_rcnn_head", FastRCNNHead(256 * 7 * 7, num_classes));
//         mask_head = register_module("mask_head", MaskHead(256, num_classes));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> forward(
//         torch::Tensor x, const std::vector<torch::Tensor>& gt_boxes = {},
//         const std::vector<torch::Tensor>& gt_classes = {}, const std::vector<torch::Tensor>& gt_masks = {}) {
//         // x: [batch, in_channels, 64, 64]
//         auto features = backbone->forward(x); // [batch, 256, 8, 8]
//
//         // RPN
//         auto [rpn_cls_logits, rpn_bbox_deltas] = rpn->forward(features); // [batch*h*w*num_anchors, 2], [batch*h*w*num_anchors, 4]
//
//         // Generate anchors
//         auto anchors = generate_anchors(features.size(2), features.size(3)).to(x.device());
//
//         // Proposal generation
//         auto proposals = generate_proposals(anchors, rpn_cls_logits, rpn_bbox_deltas, x.size(0));
//
//         // ROI Align
//         auto roi_features = roi_align->forward(features, proposals); // [num_proposals, 256, 7, 7]
//
//         // Fast R-CNN Head
//         auto [cls_logits, bbox_deltas] = fast_rcnn_head->forward(roi_features); // [num_proposals, num_classes+1], [num_proposals, (num_classes+1)*4]
//
//         // Mask Head
//         auto mask_logits = mask_head->forward(roi_features); // [num_proposals, num_classes, 14, 14]
//
//         if (!gt_boxes.empty()) {
//             // Compute losses
//             auto rpn_loss = compute_rpn_loss(rpn_cls_logits, rpn_bbox_deltas, anchors, gt_boxes);
//             auto fast_rcnn_loss = compute_fast_rcnn_loss(cls_logits, bbox_deltas, proposals, gt_boxes, gt_classes);
//             auto mask_loss = compute_mask_loss(mask_logits, proposals, gt_boxes, gt_classes, gt_masks);
//             return {rpn_loss.first, rpn_loss.second, fast_rcnn_loss.first, fast_rcnn_loss.second, mask_loss};
//         }
//
//         return {cls_logits, bbox_deltas, mask_logits, torch::Tensor(), torch::Tensor()};
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
//     torch::Tensor generate_proposals(const torch::Tensor& anchors, const torch::Tensor& cls_logits,
//                                     const torch::Tensor& bbox_deltas, int batch_size) {
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
//             // Select top 10 proposals
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
//         std::vector<int> labels;
//         std::vector<torch::Tensor> target_deltas;
//         int num_anchors = anchors.size(0) / cls_logits.size(0) * 2;
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
//         auto valid = label_tensor >= 0;
//         auto cls_loss = torch::cross_entropy(cls_logits.index_select(0, valid.nonzero().squeeze(-1)),
//                                             label_tensor.index_select(0, valid.nonzero().squeeze(-1)));
//
//         auto fg = label_tensor == 1;
//         auto reg_loss = fg.sum().item<int>() > 0 ?
//             torch::smooth_l1_loss(bbox_deltas.index_select(0, fg.nonzero().squeeze(-1)),
//                                  delta_tensor.index_select(0, fg.nonzero().squeeze(-1))) :
//             torch::tensor(0.0).to(bbox_deltas.device());
//
//         return {cls_loss, reg_loss};
//     }
//
//     std::pair<torch::Tensor, torch::Tensor> compute_fast_rcnn_loss(const torch::Tensor& cls_logits, const torch::Tensor& bbox_deltas,
//                                                                   const torch::Tensor& proposals, const std::vector<torch::Tensor>& gt_boxes,
//                                                                   const std::vector<torch::Tensor>& gt_classes) {
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
//                     target_deltas.push_back(torch::tensor({dx.item<float>(), dy.item<float>(), dw.item<float>(), dh.item<float>()}));
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
//         auto cls_loss = torch::cross_entropy(cls_logits, label_tensor);
//
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
//     torch::Tensor compute_mask_loss(const torch::Tensor& mask_logits, const torch::Tensor& proposals,
//                                    const std::vector<torch::Tensor>& gt_boxes, const std::vector<torch::Tensor>& gt_classes,
//                                    const std::vector<torch::Tensor>& gt_masks) {
//         std::vector<torch::Tensor> target_masks;
//         std::vector<int> target_labels;
//         int proposal_idx = 0;
//
//         for (int b = 0; b < gt_boxes.size(); ++b) {
//             auto b_proposals = proposals.index_select(0, (proposals.narrow(1, 0, 1) == b).nonzero().squeeze(-1));
//             if (b_proposals.size(0) == 0) continue;
//
//             auto boxes = torch::cat({
//                 b_proposals.narrow(1, 1, 1) - b_proposals.narrow(1, 3, 1) / 2,
//                 b_proposals.narrow(1, 2, 1) - b_proposals.narrow(1, 4, 1) / 2,
//                 b_proposals.narrow(1, 3, 1) - b_proposals.narrow(1, 1, 1),
//                 b_proposals.narrow(1, 4, 1) - b_proposals.narrow(1, 2, 1)
//             }, 1);
//
//             auto ious = compute_iou(boxes, gt_boxes[b]);
//             auto max_iou = ious.max(1);
//             auto max_indices = max_iou.indices();
//             auto max_values = max_iou.values();
//
//             for (int i = 0; i < boxes.size(0); ++i) {
//                 if (max_values[i].item<float>() > 0.5) {
//                     int cls = gt_classes[b][max_indices[i]].item<int>();
//                     auto gt_mask = gt_masks[b][max_indices[i]].unsqueeze(0); // [1, 64, 64]
//                     // Crop and resize to 14x14
//                     float x1 = boxes[i][0].item<float>() - boxes[i][2].item<float>() / 2;
//                     float y1 = boxes[i][1].item<float>() - boxes[i][3].item<float>() / 2;
//                     float x2 = boxes[i][0].item<float>() + boxes[i][2].item<float>() / 2;
//                     float y2 = boxes[i][1].item<float>() + boxes[i][3].item<float>() / 2;
//                     x1 = std::max(0.0f, x1 * 64);
//                     y1 = std::max(0.0f, y1 * 64);
//                     x2 = std::min(64.0f, x2 * 64);
//                     y2 = std::min(64.0f, y2 * 64);
//                     if (x2 > x1 && y2 > y1) {
//                         auto roi = gt_mask.slice(2, static_cast<int>(y1), static_cast<int>(y2))
//                                          .slice(3, static_cast<int>(x1), static_cast<int>(x2));
//                         auto resized = torch::adaptive_avg_pool2d(roi, {14, 14}).squeeze(0); // [14, 14]
//                         target_masks.push_back(resized);
//                         target_labels.push_back(cls);
//                     }
//                 }
//                 proposal_idx++;
//             }
//         }
//
//         if (target_masks.empty()) return torch::tensor(0.0).to(mask_logits.device());
//
//         auto target_mask_tensor = torch::stack(target_masks).to(mask_logits.device()); // [N, 14, 14]
//         auto target_label_tensor = torch::tensor(target_labels).to(mask_logits.device()); // [N]
//
//         auto loss = torch::zeros({}).to(mask_logits.device());
//         for (int i = 0; i < mask_logits.size(0); ++i) {
//             auto pred = mask_logits[i].index_select(0, target_label_tensor[i].unsqueeze(0)); // [1, 14, 14]
//             auto target = target_mask_tensor[i].unsqueeze(0); // [1, 14, 14]
//             loss += torch::binary_cross_entropy_with_logits(pred, target);
//         }
//
//         return loss / (target_masks.size() + 1e-6);
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
//     RoiAlign roi_align{nullptr};
//     FastRCNNHead fast_rcnn_head{nullptr};
//     MaskHead mask_head{nullptr};
// };
// TORCH_MODULE(MaskRCNN);
//
// // Instance Segmentation Dataset
// struct InstanceSegmentationDataset : torch::data::Dataset<InstanceSegmentationDataset> {
//     InstanceSegmentationDataset(const std::string& img_dir, const std::string& annot_dir, const std::string& mask_dir) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//                 std::string annot_path = annot_dir + "/" + entry.path().filename().string() + ".txt";
//                 std::string mask_path = mask_dir + "/" + entry.path().filename().string();
//                 annot_paths_.push_back(annot_path);
//                 mask_paths_.push_back(mask_path);
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
//         // Load masks
//         std::vector<torch::Tensor> masks;
//         for (int i = 0; i < classes.size(); ++i) {
//             std::string mask_path = mask_paths_[index % mask_paths_.size()];
//             cv::Mat mask = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
//             if (mask.empty()) {
//                 throw std::runtime_error("Failed to load mask: " + mask_path);
//             }
//             cv::resize(mask, mask, cv::Size(64, 64));
//             mask.convertTo(mask, CV_32F, 255 / 64.0); // Normalize to [0, 1]
//             torch::Tensor mask_tensor = torch::from_blob(mask.data, {1, mask_size, mask_size}, torch::kFloat32);
//             masks.push_back(mask_tensor);
//         }
//         torch::Tensor mask_tensor = masks.empty() ? torch::zeros({0, 64, 64}, torch::kFloat32) : torch::stack(masks);
//
//         return {img_tensor, torch::stack({class_tensor, box_tensor, mask_tensor})};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_, annot_paths_, mask_paths_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int in_channels = 1;
//         const int num_classes = 1; // Binary classification
//         const int num_anchors = 3;
//         const int batch_size = 4;
//         const float lr = 0.0; 0.001
//         const int num_epochs = 20;
//
//         // Initialize model
//         MaskRCNN model(in_channels, num_classes, num_anchors);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters()->parameters(), torch::optim::AdamOptions(lr));
//
//         // Load dataset
//         auto dataset = InstanceSegmentationDataset("./data/images", "./data/annotations", "./data/masks")
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
//                 auto images = batch->data.to(device);
//                 std::vector<torch::Tensor> gt_classes, gt_boxes, gt_masks;
//                 for (int b = 0; b < batch_size; ++b) {
//                     gt_classes.push_back(batch.target[0][b].squeeze(-1)); // [n_objects]
//                     gt_boxes.push_back(batch.target[1][b]); // [n_objects, 4]
//                     gt_masks.push_back(batch.target[2][b]); // [n_objects, 64, 64]
//                 }
//
//                 optimizer.zero_grad();
//                 auto [rpn_cls_loss, rpn_reg_loss, fast_rcnn_cls_loss, fast_rcnn_reg_loss, mask_loss] =
//                     model->forward(images, gt_boxes, gt_classes, gt_masks);
//
//                 auto loss = rpn_cls_loss + rpn_reg_loss + fast_rcnn_cls_loss + fast_rcnn_reg_loss + mask_loss;
//                 loss.backward();
//                 optimizer.step();
//
//                 loss_avg += loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Loss: " << loss.avg / batch_count << std::endl;
//
//             // Save model every 10 epochs
//             if ((epoch + 1) % 10 == 0) {
//                 torch::save(model, "maskrcnn_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "maskrcnn.pt");
//         std::cout << "Model saved as " << maskrcnn.pt" << std::endl;
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
    MaskRCNN::MaskRCNN(int num_classes, int in_channels)
    {
    }

    MaskRCNN::MaskRCNN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void MaskRCNN::reset()
    {
    }

    auto MaskRCNN::forward(std::initializer_list<std::any> tensors) -> std::any
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
