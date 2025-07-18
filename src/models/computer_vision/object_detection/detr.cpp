#include <models/computer_vision/object_detection/detr.h>


using namespace std;

//DETR GROK


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <algorithm>
//
// // CNN Backbone
// struct BackboneImpl : torch::nn::Module {
//     BackboneImpl(int in_channels) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(in_channels, 64, 3).stride(2).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 3).stride(2).padding(1)));
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
//         proj = register_module("proj", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 256, 1))); // Project to d_model
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, in_channels, 28, 28]
//         x = torch::relu(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//         x = torch::relu(bn2->forward(conv2->forward(x))); // [batch, 128, 7, 7]
//         x = torch::relu(bn3->forward(conv3->forward(x))); // [batch, 256, 4, 4]
//         x = proj->forward(x); // [batch, 256, 4, 4]
//         x = x.flatten(2).transpose(1, 2); // [batch, 16, 256]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, proj{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
// };
// TORCH_MODULE(Backbone);
//
// // Multi-Head Self-Attention (MSA)
// struct MSAImpl : torch::nn::Module {
//     MSAImpl(int embed_dim, int num_heads, float dropout = 0.1) {
//         num_heads_ = num_heads;
//         head_dim_ = embed_dim / num_heads;
//         scale_ = 1.0 / std::sqrt(head_dim_);
//         qkv = register_module("qkv", torch::nn::Linear(embed_dim, embed_dim * 3));
//         proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
//         dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, seq_len, embed_dim]
//         auto sizes = x.sizes();
//         int batch = sizes[0], seq_len = sizes[1];
//
//         // QKV projection
//         auto qkv_out = qkv->forward(x); // [batch, seq_len, embed_dim*3]
//         qkv_out = qkv_out.view({batch, seq_len, 3, num_heads_, head_dim_})
//                          .permute({2, 0, 3, 1, 4}); // [3, batch, num_heads, seq_len, head_dim]
//         auto q = qkv_out[0], k = qkv_out[1], v = qkv_out[2]; // [batch, num_heads, seq_len, head_dim]
//
//         // Attention
//         auto attn = torch::matmul(q, k.transpose(-2, -1)) * scale_; // [batch, num_heads, seq_len, seq_len]
//         attn = torch::softmax(attn, -1);
//         attn = dropout_layer->forward(attn);
//         auto out = torch::matmul(attn, v); // [batch, num_heads, seq_len, head_dim]
//         out = out.permute({0, 2, 1, 3}).contiguous()
//                  .view({batch, seq_len, num_heads_ * head_dim_}); // [batch, seq_len, embed_dim]
//         out = proj->forward(out);
//         out = dropout_layer->forward(out);
//         return out;
//     }
//
//     int num_heads_, head_dim_;
//     float scale_;
//     torch::nn::Linear qkv{nullptr}, proj{nullptr};
//     torch::nn::Dropout dropout_layer{nullptr};
// };
// TORCH_MODULE(MSA);
//
// // Multi-Head Cross-Attention
// struct CrossAttentionImpl : torch::nn::Module {
//     CrossAttentionImpl(int embed_dim, int num_heads, float dropout = 0.1) {
//         num_heads_ = num_heads;
//         head_dim_ = embed_dim / num_heads;
//         scale_ = 1.0 / std::sqrt(head_dim_);
//         q = register_module("q", torch::nn::Linear(embed_dim, embed_dim));
//         kv = register_module("kv", torch::nn::Linear(embed_dim, embed_dim * 2));
//         proj = register_module("proj", torch::nn::Linear(embed_dim, embed_dim));
//         dropout_layer = register_module("dropout", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor queries, torch::Tensor keys) {
//         // queries: [batch, n_queries, embed_dim], keys: [batch, seq_len, embed_dim]
//         int batch = queries.size(0), n_queries = queries.size(1);
//
//         // Query projection
//         auto q_out = q->forward(queries); // [batch, n_queries, embed_dim]
//         q_out = q_out.view({batch, n_queries, num_heads_, head_dim_})
//                      .permute({0, 2, 1, 3}); // [batch, num_heads, n_queries, head_dim]
//
//         // Key/Value projection
//         auto kv_out = kv->forward(keys); // [batch, seq_len, embed_dim*2]
//         kv_out = kv_out.view({batch, -1, 2, num_heads_, head_dim_})
//                        .permute({2, 0, 3, 1, 4}); // [2, batch, num_heads, seq_len, head_dim]
//         auto k = kv_out[0], v = kv_out[1]; // [batch, num_heads, seq_len, head_dim]
//
//         // Attention
//         auto attn = torch::matmul(q_out, k.transpose(-2, -1)) * scale_; // [batch, num_heads, n_queries, seq_len]
//         attn = torch::softmax(attn, -1);
//         attn = dropout_layer->forward(attn);
//         auto out = torch::matmul(attn, v); // [batch, num_heads, n_queries, head_dim]
//         out = out.permute({0, 2, 1, 3}).contiguous()
//                  .view({batch, n_queries, num_heads_ * head_dim_}); // [batch, n_queries, embed_dim]
//         out = proj->forward(out);
//         out = dropout_layer->forward(out);
//         return out;
//     }
//
//     int num_heads_, head_dim_;
//     float scale_;
//     torch::nn::Linear q{nullptr}, kv{nullptr}, proj{nullptr};
//     torch::nn::Dropout dropout_layer{nullptr};
// };
// TORCH_MODULE(CrossAttention);
//
// // Feed-Forward Network (FFN)
// struct FFNImpl : torch::nn::Module {
//     FFNImpl(int embed_dim, int mlp_ratio = 4, float dropout = 0.1) {
//         fc1 = register_module("fc1", torch::nn::Linear(embed_dim, embed_dim * mlp_ratio));
//         fc2 = register_module("fc2", torch::nn::Linear(embed_dim * mlp_ratio, embed_dim));
//         dropout1 = register_module("dropout1", torch::nn::Dropout(dropout));
//         dropout2 = register_module("dropout2", torch::nn::Dropout(dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = fc1->forward(x);
//         x = torch::relu(x);
//         x = dropout1->forward(x);
//         x = fc2->forward(x);
//         x = dropout2->forward(x);
//         return x;
//     }
//
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//     torch::nn::Dropout dropout1{nullptr}, dropout2{nullptr};
// };
// TORCH_MODULE(FFN);
//
// // Transformer Encoder Layer
// struct EncoderLayerImpl : torch::nn::Module {
//     EncoderLayerImpl(int embed_dim, int num_heads, int mlp_ratio = 4, float dropout = 0.1) {
//         norm1 = register_module("norm1", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         attn = register_module("attn", MSA(embed_dim, num_heads, dropout));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         ffn = register_module("ffn", FFN(embed_dim, mlp_ratio, dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto residual = x;
//         x = norm1->forward(x);
//         x = attn->forward(x);
//         x = residual + x;
//         residual = x;
//         x = norm2->forward(x);
//         x = ffn->forward(x);
//         x = residual + x;
//         return x;
//     }
//
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
//     MSA attn{nullptr};
//     FFN ffn{nullptr};
// };
// TORCH_MODULE(EncoderLayer);
//
// // Transformer Decoder Layer
// struct DecoderLayerImpl : torch::nn::Module {
//     DecoderLayerImpl(int embed_dim, int num_heads, int mlp_ratio = 4, float dropout = 0.1) {
//         norm1 = register_module("norm1", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         self_attn = register_module("self_attn", MSA(embed_dim, num_heads, dropout));
//         norm2 = register_module("norm2", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         cross_attn = register_module("cross_attn", CrossAttention(embed_dim, num_heads, dropout));
//         norm3 = register_module("norm3", torch::nn::LayerNorm(
//             torch::nn::LayerNormOptions({embed_dim}).eps(1e-6)));
//         ffn = register_module("ffn", FFN(embed_dim, mlp_ratio, dropout));
//     }
//
//     torch::Tensor forward(torch::Tensor queries, torch::Tensor memory) {
//         // queries: [batch, n_queries, embed_dim], memory: [batch, seq_len, embed_dim]
//         auto residual = queries;
//         queries = norm1->forward(queries);
//         queries = self_attn->forward(queries);
//         queries = residual + queries;
//         residual = queries;
//         queries = norm2->forward(queries);
//         queries = cross_attn->forward(queries, memory);
//         queries = residual + queries;
//         residual = queries;
//         queries = norm3->forward(queries);
//         queries = ffn->forward(queries);
//         queries = residual + queries;
//         return queries;
//     }
//
//     torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
//     MSA self_attn{nullptr};
//     CrossAttention cross_attn{nullptr};
//     FFN ffn{nullptr};
// };
// TORCH_MODULE(DecoderLayer);
//
// // DETR Model
// struct DETRImpl : torch::nn::Module {
//     DETRImpl(int in_channels, int num_classes, int n_queries, int embed_dim, int num_heads, int num_encoder_layers, int num_decoder_layers) {
//         num_classes_ = num_classes;
//         n_queries_ = n_queries;
//         backbone = register_module("backbone", Backbone(in_channels));
//         pos_embed = register_parameter("pos_embed", torch::randn({1, 16, embed_dim})); // For 4x4 feature map
//         query_embed = register_parameter("query_embed", torch::randn({n_queries, embed_dim}));
//
//         // Encoder
//         for (int i = 0; i < num_encoder_layers; ++i) {
//             encoder->push_back(EncoderLayer(embed_dim, num_heads));
//             register_module("encoder_layer_" + std::to_string(i), encoder[i]);
//         }
//
//         // Decoder
//         for (int i = 0; i < num_decoder_layers; ++i) {
//             decoder->push_back(DecoderLayer(embed_dim, num_heads));
//             register_module("decoder_layer_" + std::to_string(i), decoder[i]);
//         }
//
//         // Prediction heads
//         class_head = register_module("class_head", torch::nn::Linear(embed_dim, num_classes + 1)); // +1 for no-object
//         box_head = register_module("box_head", torch::nn::Linear(embed_dim, 4)); // [cx, cy, w, h]
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // x: [batch, in_channels, 28, 28]
//         int batch = x.size(0);
//
//         // Backbone
//         auto features = backbone->forward(x); // [batch, 16, embed_dim]
//         features = features + pos_embed; // Add positional embedding
//
//         // Encoder
//         auto memory = features;
//         for (auto& layer : *encoder) {
//             memory = layer->forward(memory); // [batch, 16, embed_dim]
//         }
//
//         // Decoder
//         auto queries = query_embed.unsqueeze(0).expand({batch, -1, -1}); // [batch, n_queries, embed_dim]
//         auto output = queries;
//         for (auto& layer : *decoder) {
//             output = layer->forward(output, memory); // [batch, n_queries, embed_dim]
//         }
//
//         // Prediction
//         auto class_logits = class_head->forward(output); // [batch, n_queries, num_classes+1]
//         auto boxes = torch::sigmoid(box_head->forward(output)); // [batch, n_queries, 4]
//         return {class_logits, boxes};
//     }
//
//     int num_classes_, n_queries_;
//     Backbone backbone{nullptr};
//     torch::nn::ModuleList encoder{nullptr}, decoder{nullptr};
//     torch::nn::Linear class_head{nullptr}, box_head{nullptr};
//     torch::Tensor pos_embed, query_embed;
// };
// TORCH_MODULE(DETR);
//
// // Hungarian Matcher (Simplified)
// struct HungarianMatcher {
//     static std::vector<std::pair<int, int>> match(const torch::Tensor& pred_logits, const torch::Tensor& pred_boxes,
//                                                   const torch::Tensor& gt_classes, const torch::Tensor& gt_boxes) {
//         // pred_logits: [batch, n_queries, num_classes+1], pred_boxes: [batch, n_queries, 4]
//         // gt_classes: [batch, n_gt, 1], gt_boxes: [batch, n_gt, 4]
//         int batch = pred_logits.size(0);
//         int n_queries = pred_logits.size(1);
//         std::vector<std::pair<int, int>> indices;
//
//         for (int b = 0; b < batch; ++b) {
//             int n_gt = gt_classes[b].size(0);
//             auto costs = torch::zeros({n_queries, n_gt}).to(pred_logits.device());
//
//             // Classification cost
//             auto probs = torch::softmax(pred_logits[b], -1); // [n_queries, num_classes+1]
//             for (int i = 0; i < n_gt; ++i) {
//                 int cls = gt_classes[b][i].item<int>();
//                 costs.slice(1, i, i+1) = -probs.slice(1, cls, cls+1);
//             }
//
//             // Box cost (L1 distance)
//             auto box_diff = torch::abs(pred_boxes[b].unsqueeze(1) - gt_boxes[b].unsqueeze(0)); // [n_queries, n_gt, 4]
//             costs += box_diff.sum(-1); // [n_queries, n_gt]
//
//             // Simple greedy matching
//             std::vector<bool> used(n_gt, false);
//             for (int i = 0; i < n_queries; ++i) {
//                 if (i >= n_gt) break;
//                 auto cost = costs[i];
//                 auto min_idx = std::distance(cost.data_ptr<float>(),
//                     std::min_element(cost.data_ptr<float>(), cost.data_ptr<float>() + n_gt));
//                 if (!used[min_idx]) {
//                     indices.emplace_back(i, min_idx);
//                     used[min_idx] = true;
//                 }
//             }
//         }
//
//         return indices;
//     }
// };
//
// // DETR Loss
// struct DETRLossImpl : torch::nn::Module {
//     DETRLossImpl(float lambda_class = 1.0, float lambda_l1 = 5.0, float lambda_giou = 2.0)
//         : lambda_class_(lambda_class), lambda_l1_(lambda_l1), lambda_giou_(lambda_giou) {}
//
//     torch::Tensor compute_giou(const torch::Tensor& boxes1, const torch::Tensor& boxes2) {
//         // boxes1, boxes2: [N, 4] (cx, cy, w, h)
//         auto x1 = boxes1.narrow(1, 0, 1) - boxes1.narrow(1, 2, 1) / 2; // [N, 1]
//         auto y1 = boxes1.narrow(1, 1, 1) - boxes1.narrow(1, 3, 1) / 2;
//         auto x2 = boxes1.narrow(1, 0, 1) + boxes1.narrow(1, 2, 1) / 2;
//         auto y2 = boxes1.narrow(1, 1, 1) + boxes1.narrow(1, 3, 1) / 2;
//
//         auto x1g = boxes2.narrow(1, 0, 1) - boxes2.narrow(1, 2, 1) / 2;
//         auto y1g = boxes2.narrow(1, 1, 1) - boxes2.narrow(1, 3, 1) / 2;
//         auto x2g = boxes2.narrow(1, 0, 1) + boxes2.narrow(1, 2, 1) / 2;
//         auto y2g = boxes2.narrow(1, 1, 1) + boxes2.narrow(1, 3, 1) / 2;
//
//         // Intersection
//         auto xi1 = torch::maximum(x1, x1g); // [N, 1]
//         auto yi1 = torch::maximum(y1, y1g);
//         auto xi2 = torch::minimum(x2, x2g);
//         auto yi2 = torch::minimum(y2, y2g);
//         auto inter_area = torch::clamp(xi2 - xi1, 0) * torch::clamp(yi2 - yi1, 0); // [N, 1]
//
//         // Union
//         auto area1 = boxes1.narrow(1, 2, 1) * boxes1.narrow(1, 3, 1); // [N, 1]
//         auto area2 = boxes2.narrow(1, 2, 1) * boxes2.narrow(1, 3, 1);
//         auto union_area = area1 + area2 - inter_area;
//
//         // Enclosing box
//         auto xe1 = torch::minimum(x1, x1g);
//         auto ye1 = torch::minimum(y1, y1g);
//         auto xe2 = torch::maximum(x2, x2g);
//         auto ye2 = torch::maximum(y2, y2g);
//         auto encl_area = (xe2 - xe1) * (ye2 - ye1);
//
//         auto iou = inter_area / union_area;
//         auto giou = iou - (encl_area - union_area) / encl_area;
//         return giou.squeeze(-1); // [N]
//     }
//
//     torch::Tensor forward(const torch::Tensor& pred_logits, const torch::Tensor& pred_boxes,
//                          const torch::Tensor& gt_classes, const torch::Tensor& gt_boxes) {
//         // pred_logits: [batch, n_queries, num_classes+1], pred_boxes: [batch, n_queries, 4]
//         // gt_classes: [batch, n_gt, 1], gt_boxes: [batch, n_gt, 4]
//         int batch = pred_logits.size(0);
//         int n_queries = pred_logits.size(1);
//
//         auto indices = HungarianMatcher::match(pred_logits, pred_boxes, gt_classes, gt_boxes);
//
//         torch::Tensor loss_class = torch::tensor(0.0).to(pred_logits.device());
//         torch::Tensor loss_boxes = torch::tensor(0.0).to(pred_logits.device());
//         torch::Tensor loss_giou = torch::tensor(0.0).to(pred_logits.device());
//         int count = 0;
//
//         auto ce_loss = torch::nn::CrossEntropyLoss();
//
//         for (int b = 0; b < batch; ++b) {
//             std::vector<int> pred_idx, gt_idx;
//             for (const auto& idx : indices) {
//                 pred_idx.push_back(idx.first);
//                 gt_idx.push_back(idx.second);
//             }
//
//             if (!pred_idx.empty()) {
//                 auto pred_cls = pred_logits[b].index_select(0, torch::tensor(pred_idx).to(pred_logits.device())); // [n_matched, num_classes+1]
//                 auto gt_cls = gt_classes[b].index_select(0, torch::tensor(gt_idx).to(pred_logits.device())).squeeze(-1); // [n_matched]
//                 loss_class += ce_loss->forward(pred_cls, gt_cls);
//
//                 auto pred_bx = pred_boxes[b].index_select(0, torch::tensor(pred_idx).to(pred_logits.device())); // [n_matched, 4]
//                 auto gt_bx = gt_boxes[b].index_select(0, torch::tensor(gt_idx).to(pred_logits.device())); // [n_matched, 4]
//                 loss_boxes += torch::abs(pred_bx - gt_bx).sum() / pred_bx.size(0);
//
//                 auto giou = compute_giou(pred_bx, gt_bx); // [n_matched]
//                 loss_giou += (1 - giou).sum() / pred_bx.size(0);
//
//                 count++;
//             }
//
//             // Background (no-object) loss
//             std::vector<int> bg_idx;
//             for (int i = 0; i < n_queries; ++i) {
//                 if (std::find(pred_idx.begin(), pred_idx.end(), i) == pred_idx.end()) {
//                     bg_idx.push_back(i);
//                 }
//             }
//             if (!bg_idx.empty()) {
//                 auto bg_cls = pred_logits[b].index_select(0, torch::tensor(bg_idx).to(pred_logits.device())); // [n_bg, num_classes+1]
//                 auto bg_target = torch::full({bg_idx.size()}, pred_logits.size(-1) - 1, torch::kLong).to(pred_logits.device()); // No-object class
//                 loss_class += ce_loss->forward(bg_cls, bg_target);
//                 count++;
//             }
//         }
//
//         if (count > 0) {
//             loss_class /= count;
//             loss_boxes /= count;
//             loss_giou /= count;
//         }
//
//         return lambda_class_ * loss_class + lambda_l1_ * loss_boxes + lambda_giou_ * loss_giou;
//     }
//
//     float lambda_class_, lambda_l1_, lambda_giou_;
// };
// TORCH_MODULE(DETRLoss);
//
// // Dataset for Images and Annotations
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
//         const int num_classes = 1; // Binary detection (e.g., digit vs. no-object)
//         const int n_queries = 5;
//         const int embed_dim = 256;
//         const int num_heads = 8;
//         const int num_encoder_layers = 3;
//         const int num_decoder_layers = 3;
//         const int batch_size = 4;
//         const float lr = 0.0001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         DETR model(in_channels, num_classes, n_queries, embed_dim, num_heads, num_encoder_layers, num_decoder_layers);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Loss function
//         DETRLoss loss_fn(1.0, 5.0, 2.0);
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
//                 auto targets = batch.target.to(device);
//                 auto gt_classes = targets[0]; // [batch, n_gt, 1]
//                 auto gt_boxes = targets[1]; // [batch, n_gt, 4]
//
//                 optimizer.zero_grad();
//                 auto [pred_logits, pred_boxes] = model->forward(images); // [batch, n_queries, num_classes+1], [batch, n_queries, 4]
//                 auto loss = loss_fn.forward(pred_logits, pred_boxes, gt_classes, gt_boxes);
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
//                 torch::save(model, "detr_epoch_" + std::to_string(epoch + 1) + ".pt");
//             }
//         }
//
//         // Save final model
//         torch::save(model, "detr.pt");
//         std::cout << "Model saved as detr.pt" << std::endl;
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
    DETR::DETR(int num_classes, int in_channels)
    {
    }

    DETR::DETR(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DETR::reset()
    {
    }

    auto DETR::forward(std::initializer_list<std::any> tensors) -> std::any
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
