#include "include/models/computer_vision/object_detection/yolox.h"


using namespace std;
//YOLOX GROK


// #include <torch/torch.h>
// #include <torch/script.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <random>
//
// // Simplified CSP-Darknet-53 Backbone
// struct BackboneImpl : torch::nn::Module {
//     BackboneImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1)));
//         conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = relu->forward(conv1->forward(x)); // [batch, 16, 28, 28]
//         x = relu->forward(conv2->forward(x)); // [batch, 32, 14, 14]
//         x = relu->forward(conv3->forward(x)); // [batch, 64, 7, 7]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(Backbone);
//
// // Simplified FPN Neck
// struct NeckImpl : torch::nn::Module {
//     NeckImpl() {
//         lateral_conv = register_module("lateral_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 1).stride(1)));
//         upsample = register_module("upsample", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(2).mode(torch::kNearest)));
//         fpn_conv = register_module("fpn_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 3).stride(1).padding(1)));
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = relu->forward(lateral_conv->forward(x)); // [batch, 32, 7, 7]
//         x = upsample->forward(x); // [batch, 32, 14, 14]
//         x = relu->forward(fpn_conv->forward(x)); // [batch, 16, 14, 14]
//         return x;
//     }
//
//     torch::nn::Conv2d lateral_conv{nullptr}, fpn_conv{nullptr};
//     torch::nn::Upsample upsample{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(Neck);
//
// // Decoupled Head for YOLOX
// struct HeadImpl : torch::nn::Module {
//     HeadImpl(int num_classes) : num_classes_(num_classes) {
//         cls_conv = register_module("cls_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, num_classes, 1).stride(1)));
//         reg_conv = register_module("reg_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 4, 1).stride(1))); // x, y, w, h
//         obj_conv = register_module("obj_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 1, 1).stride(1))); // objectness
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         auto cls_pred = torch::sigmoid(cls_conv->forward(x)); // [batch, num_classes, 14, 14]
//         auto reg_pred = reg_conv->forward(x); // [batch, 4, 14, 14]
//         // Apply exp to w, h for positive values
//         reg_pred.slice(1, 2, 4) = torch::exp(reg_pred.slice(1, 2, 4));
//         auto obj_pred = torch::sigmoid(obj_conv->forward(x)); // [batch, 1, 14, 14]
//         return {cls_pred, reg_pred, obj_pred};
//     }
//
//     int num_classes_;
//     torch::nn::Conv2d cls_conv{nullptr}, reg_conv{nullptr}, obj_conv{nullptr};
// };
// TORCH_MODULE(Head);
//
// // Simplified YOLOX Model
// struct YOLOXImpl : torch::nn::Module {
//     YOLOXImpl(int num_classes) {
//         backbone = register_module("backbone", Backbone());
//         neck = register_module("neck", Neck());
//         head = register_module("head", Head(num_classes));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         x = backbone->forward(x);
//         x = neck->forward(x);
//         return head->forward(x);
//     }
//
//     Backbone backbone{nullptr};
//     Neck neck{nullptr};
//     Head head{nullptr};
// };
// TORCH_MODULE(YOLOX);
//
// // YOLOX Loss with SimOTA
// struct YOLOXLossImpl : torch::nn::Module {
//     YOLOXLossImpl(int num_classes, float lambda_iou = 5.0)
//         : num_classes_(num_classes), lambda_iou_(lambda_iou) {}
//
//     torch::Tensor forward(const std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>& pred,
//                          torch::Tensor target) {
//         auto [cls_pred, reg_pred, obj_pred] = pred;
//         auto batch_size = cls_pred.size(0);
//         auto grid_size = cls_pred.size(2); // 14x14
//         auto device = cls_pred.device();
//
//         torch::Tensor loss_cls = torch::zeros({}, device);
//         torch::Tensor loss_reg = torch::zeros({}, device);
//         torch::Tensor loss_obj = torch::zeros({}, device);
//
//         // Generate grid offsets
//         auto grid_x = torch::arange(grid_size, torch::kFloat32, device).repeat({grid_size, 1});
//         auto grid_y = grid_x.transpose(0, 1);
//         grid_x = grid_x.view({1, grid_size, grid_size, 1});
//         grid_y = grid_y.view({1, grid_size, grid_size, 1});
//
//         // Simplified SimOTA: Assign each GT to the grid cell with highest IoU
//         for (int b = 0; b < batch_size; ++b) {
//             auto target_b = target[b]; // [max_objects, 5] (x, y, w, h, class)
//             auto cls_pred_b = cls_pred[b]; // [num_classes, 14, 14]
//             auto reg_pred_b = reg_pred[b]; // [4, 14, 14]
//             auto obj_pred_b = obj_pred[b]; // [1, 14, 14]
//
//             for (int t = 0; t < target_b.size(0); ++t) {
//                 if (target_b[t][4].item<float>() < 0) continue; // Invalid object
//                 float tx = target_b[t][0].item<float>() * grid_size;
//                 float ty = target_b[t][1].item<float>() * grid_size;
//                 float tw = target_b[t][2].item<float>() * grid_size;
//                 float th = target_b[t][3].item<float>() * grid_size;
//                 int class_id = target_b[t][4].item<float>();
//                 int gx = static_cast<int>(tx);
//                 int gy = static_cast<int>(ty);
//                 if (gx >= grid_size || gy >= grid_size || gx < 0 || gy < 0) continue;
//
//                 // Predicted box at grid cell
//                 auto pred_x = reg_pred_b[0][gy][gx] + gx;
//                 auto pred_y = reg_pred_b[1][gy][gx] + gy;
//                 auto pred_w = reg_pred_b[2][gy][gx];
//                 auto pred_h = reg_pred_b[3][gy][gx];
//
//                 // Compute IoU loss
//                 float inter_x1 = std::max(pred_x.item<float>() - pred_w.item<float>() / 2, tx - tw / 2);
//                 float inter_y1 = std::max(pred_y.item<float>() - pred_h.item<float>() / 2, ty - th / 2);
//                 float inter_x2 = std::min(pred_x.item<float>() + pred_w.item<float>() / 2, tx + tw / 2);
//                 float inter_y2 = std::min(pred_y.item<float>() + pred_h.item<float>() / 2, ty + th / 2);
//                 float inter_area = std::max(0.0f, inter_x2 - inter_x1) * std::max(0.0f, inter_y2 - inter_y1);
//                 float pred_area = pred_w.item<float>() * pred_h.item<float>();
//                 float gt_area = tw * th;
//                 float iou = inter_area / (pred_area + gt_area - inter_area + 1e-6);
//                 loss_reg += lambda_iou_ * (1.0 - iou);
//
//                 // Classification loss
//                 auto target_cls = torch::zeros({num_classes_}, torch::kFloat32, device);
//                 target_cls[class_id] = 1.0;
//                 loss_cls += torch::nn::functional::binary_cross_entropy(
//                     cls_pred_b.slice(0, 0, num_classes_).select(1, gy).select(1, gx), target_cls);
//
//                 // Objectness loss
//                 loss_obj += torch::nn::functional::binary_cross_entropy(
//                     obj_pred_b[0][gy][gx], torch::tensor(1.0, torch::kFloat32, device));
//             }
//
//             // Negative samples (no object)
//             auto noobj_mask = torch::ones({grid_size, grid_size}, torch::kBool, device);
//             for (int t = 0; t < target_b.size(0); ++t) {
//                 if (target_b[t][4].item<float>() < 0) continue;
//                 int gx = static_cast<int>(target_b[t][0].item<float>() * grid_size);
//                 int gy = static_cast<int>(target_b[t][1].item<float>() * grid_size);
//                 if (gx >= 0 && gx < grid_size && gy >= 0 && gy < grid_size) {
//                     noobj_mask[gy][gx] = false;
//                 }
//             }
//             loss_obj += torch::nn::functional::binary_cross_entropy(
//                 obj_pred_b[0].masked_select(noobj_mask), torch::zeros_like(obj_pred_b[0].masked_select(noobj_mask)));
//         }
//
//         return (loss_cls + loss_reg + loss_obj) / batch_size;
//     }
//
//     int num_classes_;
//     float lambda_iou_;
// };
// TORCH_MODULE(YOLOXLoss);
//
// // Custom Dataset
// struct YOLODataset : torch::data::Dataset<YOLODataset> {
//     YOLODataset(const std::string& img_dir, const std::string& label_dir, int max_objects = 10)
//         : max_objects_(max_objects) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//                 std::string label_path = label_dir + "/" + entry.path().stem().string() + ".txt";
//                 label_paths_.push_back(label_path);
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         cv::Mat image = cv::imread(image_paths_[index], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index]);
//         }
//         image.convertTo(image, CV_32F, 1.0 / 255.0);
//         torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//         torch::Tensor label_tensor = torch::zeros({max_objects_, 5}, torch::kFloat32);
//         label_tensor.fill_(-1);
//         std::ifstream infile(label_paths_[index]);
//         if (infile.is_open()) {
//             std::string line;
//             int obj_idx = 0;
//             while (std::getline(infile, line) && obj_idx < max_objects_) {
//                 std::istringstream iss(line);
//                 float cls, x, y, w, h;
//                 if (iss >> cls >> x >> y >> w >> h) {
//                     label_tensor[obj_idx] = torch::tensor({x, y, w, h, cls}, torch::kFloat32);
//                     obj_idx++;
//                 }
//             }
//             infile.close();
//         }
//
//         return {img_tensor, label_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_, label_paths_;
//     int max_objects_;
// };
//
// // Load Pre-trained Weights
// bool LoadPretrainedWeights(YOLOX& model, const std::string& pretrained_path, torch::Device device) {
//     try {
//         torch::jit::script::Module module = torch::jit::load(pretrained_path, device);
//         module.eval();
//         torch::OrderedDict<std::string, torch::Tensor> pretrained_dict = module.named_parameters();
//         auto model_dict = model->named_parameters();
//
//         for (const auto& pair : pretrained_dict) {
//             if (model_dict.contains(pair.key())) {
//                 if (model_dict[pair.key()].sizes() == pair.value().sizes()) {
//                     model_dict[pair.key()].copy_(pair.value());
//                     std::cout << "Loaded parameter: " << pair.key() << std::endl;
//                 } else {
//                     std::cerr << "Size mismatch for parameter: " << pair.key() << std::endl;
//                 }
//             } else {
//                 std::cerr << "Parameter not found in model: " << pair.key() << std::endl;
//             }
//         }
//         return true;
//     } catch (const std::exception& e) {
//         std::cerr << "Error loading pre-trained weights: " << e.what() << std::endl;
//         return false;
//     }
// }
//
// int main(int argc, char* argv[]) {
//     if (argc != 2) {
//         std::cerr << "Usage: " << argv[0] << " <pretrained_model.pt>" << std::endl;
//         return -1;
//     }
//
//     try {
//         // Device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Initialize model and loss
//         int num_classes = 2; // Example: person, car
//         YOLOX model(num_classes);
//         YOLOXLoss loss(num_classes);
//         model->to(device);
//         loss->to(device);
//
//         // Load pre-trained weights
//         if (!LoadPretrainedWeights(model, argv[1], device)) {
//             std::cerr << "Failed to load pre-trained weights. Exiting." << std::endl;
//             return -1;
//         }
//         std::cout << "Pre-trained weights loaded successfully." << std::endl;
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.0001));
//
//         // Dataset
//         auto dataset = YOLODataset("./data/images", "./data/labels", 10)
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(32).workers(2));
//
//         // Training loop
//         model->train();
//         for (int epoch = 0; epoch < 10; ++epoch) {
//             float total_loss = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto images = batch.data.to(device);
//                 auto labels = batch.target.to(device);
//
//                 optimizer.zero_grad();
//                 auto output = model->forward(images);
//                 auto loss_value = loss->forward(output, labels);
//                 loss_value.backward();
//                 optimizer.step();
//
//                 total_loss += loss_value.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
//         }
//
//         // Save fine-tuned model
//         torch::save(model, "yolox_finetuned.pt");
//         std::cout << "Fine-tuned model saved as yolox_finetuned.pt" << std::endl;
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
    YoloX::YoloX(int num_classes, int in_channels)
    {
    }

    YoloX::YoloX(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void YoloX::reset()
    {
    }

    auto YoloX::forward(std::initializer_list<std::any> tensors) -> std::any
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
