#include <models/computer_vision/object_detection/yolo.h>


using namespace std;
//YOLO10 GROK

//
//#include <torch/torch.h>
//#include <opencv2/opencv.hpp>
//#include <opencv2/imgproc.hpp>
//#include <filesystem>
//#include <iostream>
//#include <vector>
//#include <string>
//#include <fstream>
//#include <random>
//
//// Simplified YOLOv10 Backbone
//struct BackboneImpl : torch::nn::Module {
//    BackboneImpl() {
//        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1)));
//        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1)));
//        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)));
//        relu = register_module("relu", torch::nn::ReLU());
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        x = relu->forward(conv1->forward(x)); // [batch, 16, 28, 28]
//        x = relu->forward(conv2->forward(x)); // [batch, 32, 14, 14]
//        x = relu->forward(conv3->forward(x)); // [batch, 64, 7, 7]
//        return x;
//    }
//
//    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//    torch::nn::ReLU relu{nullptr};
//};
//TORCH_MODULE(Backbone);
//
//// Simplified YOLOv10 Neck
//struct NeckImpl : torch::nn::Module {
//    NeckImpl() {
//        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 1).stride(1)));
//        upsample = register_module("upsample", torch::nn::Upsample(torch::nn::UpsampleOptions().scale_factor(2).mode(torch::kNearest)));
//        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 3).stride(1).padding(1)));
//        relu = register_module("relu", torch::nn::ReLU());
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        x = relu->forward(conv1->forward(x)); // [batch, 32, 7, 7]
//        x = upsample->forward(x); // [batch, 32, 14, 14]
//        x = relu->forward(conv2->forward(x)); // [batch, 16, 14, 14]
//        return x;
//    }
//
//    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
//    torch::nn::Upsample upsample{nullptr};
//    torch::nn::ReLU relu{nullptr};
//};
//TORCH_MODULE(Neck);
//
//// Simplified YOLOv10 Head
//struct HeadImpl : torch::nn::Module {
//    HeadImpl(int num_classes, int num_anchors = 3) : num_classes_(num_classes), num_anchors_(num_anchors) {
//        // Output: [x, y, w, h, conf, classes] per anchor
//        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, num_anchors * (5 + num_classes), 1).stride(1)));
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        x = conv->forward(x); // [batch, num_anchors * (5 + num_classes), 14, 14]
//        // Reshape to [batch, num_anchors, 5 + num_classes, grid_h, grid_w]
//        auto batch_size = x.size(0);
//        x = x.view({batch_size, num_anchors_, 5 + num_classes_, 14, 14});
//        x = x.permute({0, 1, 3, 4, 2}); // [batch, num_anchors, grid_h, grid_w, 5 + num_classes]
//        // Apply sigmoid to x, y, conf, and class probs
//        x.select(4, 0) = torch::sigmoid(x.select(4, 0)); // x
//        x.select(4, 1) = torch::sigmoid(x.select(4, 1)); // y
//        x.select(4, 4) = torch::sigmoid(x.select(4, 4)); // conf
//        for (int i = 5; i < 5 + num_classes_; ++i) {
//            x.select(4, i) = torch::sigmoid(x.select(4, i)); // class probs
//        }
//        return x;
//    }
//
//    int num_classes_, num_anchors_;
//    torch::nn::Conv2d conv{nullptr};
//};
//TORCH_MODULE(Head);
//
//// Simplified YOLOv10 Model
//struct YOLOv10Impl : torch::nn::Module {
//    YOLOv10Impl(int num_classes, int num_anchors = 3) {
//        backbone = register_module("backbone", Backbone());
//        neck = register_module("neck", Neck());
//        head = register_module("head", Head(num_classes, num_anchors));
//    }
//
//    torch::Tensor forward(torch::Tensor x) {
//        x = backbone->forward(x);
//        x = neck->forward(x);
//        x = head->forward(x);
//        return x;
//    }
//
//    Backbone backbone{nullptr};
//    Neck neck{nullptr};
//    Head head{nullptr};
//};
//TORCH_MODULE(YOLOv10);
//
//// YOLO Loss Function
//struct YOLOLossImpl : torch::nn::Module {
//    YOLOLossImpl(int num_classes, int num_anchors, float lambda_coord = 5.0, float lambda_noobj = 0.5)
//            : num_classes_(num_classes), num_anchors_(num_anchors), lambda_coord_(lambda_coord), lambda_noobj_(lambda_noobj) {
//        // Predefined anchors (width, height) scaled to grid
//        anchors_ = torch::tensor({{1.0, 1.0}, {2.0, 2.0}, {0.5, 0.5}}, torch::kFloat32);
//    }
//
//    torch::Tensor forward(torch::Tensor pred, torch::Tensor target) {
//        auto batch_size = pred.size(0);
//        auto grid_size = pred.size(2); // 14x14
//        auto device = pred.device();
//
//        // Initialize losses
//        torch::Tensor loss_xy = torch::zeros({}, device);
//        torch::Tensor loss_wh = torch::zeros({}, device);
//        torch::Tensor loss_conf = torch::zeros({}, device);
//        torch::Tensor loss_class = torch::zeros({}, device);
//
//        // Generate grid offsets
//        auto grid_x = torch::arange(grid_size, torch::kFloat32, device).repeat({grid_size, 1});
//        auto grid_y = grid_x.transpose(0, 1);
//        grid_x = grid_x.view({1, 1, grid_size, grid_size, 1});
//        grid_y = grid_y.view({1, 1, grid_size, grid_size, 1});
//
//        // Process each batch
//        for (int b = 0; b < batch_size; ++b) {
//            auto pred_b = pred[b]; // [num_anchors, grid_h, grid_w, 5 + num_classes]
//            auto target_b = target[b]; // [max_objects, 5] (x, y, w, h, class)
//
//            // Create target tensors
//            auto obj_mask = torch::zeros({num_anchors_, grid_size, grid_size}, torch::kBool, device);
//            auto noobj_mask = torch::ones({num_anchors_, grid_size, grid_size}, torch::kBool, device);
//            auto target_xy = torch::zeros({num_anchors_, grid_size, grid_size, 2}, torch::kFloat32, device);
//            auto target_wh = torch::zeros({num_anchors_, grid_size, grid_size, 2}, torch::kFloat32, device);
//            auto target_conf = torch::zeros({num_anchors_, grid_size, grid_size}, torch::kFloat32, device);
//            auto target_class = torch::zeros({num_anchors_, grid_size, grid_size, num_classes_}, torch::kFloat32, device);
//
//            // Assign ground truth to grid cells
//            for (int t = 0; t < target_b.size(0); ++t) {
//                if (target_b[t][4].item<float>() < 0) continue; // Invalid object
//                float tx = target_b[t][0].item<float>() * grid_size;
//                float ty = target_b[t][1].item<float>() * grid_size;
//                float tw = target_b[t][2].item<float>() * grid_size;
//                float th = target_b[t][3].item<float>() * grid_size;
//                int class_id = target_b[t][4].item<float>();
//                int gx = static_cast<int>(tx);
//                int gy = static_cast<int>(ty);
//                if (gx >= grid_size || gy >= grid_size || gx < 0 || gy < 0) continue;
//
//                // Compute IoU with anchors to select best anchor
//                auto anchor_boxes = anchors_.clone().to(device);
//                auto gt_box = torch::tensor({tw, th}, torch::kFloat32, device).unsqueeze(0);
//                auto anchor_w = anchor_boxes.select(1, 0);
//                auto anchor_h = anchor_boxes.select(1, 1);
//                auto inter_w = torch::min(anchor_w, gt_box[0][0]);
//                auto inter_h = torch::min(anchor_h, gt_box[0][1]);
//                auto inter_area = inter_w * inter_h;
//                auto union_area = anchor_w * anchor_h + gt_box[0][0] * gt_box[0][1] - inter_area;
//                auto iou = inter_area / (union_area + 1e-6);
//                auto [_, best_anchor] = torch::max(iou, 0);
//
//                // Assign target
//                obj_mask[best_anchor.item<int>()][gy][gx] = true;
//                noobj_mask[best_anchor.item<int>()][gy][gx] = false;
//                target_xy[best_anchor.item<int>()][gy][gx] = torch::tensor({tx - gx, ty - gy}, torch::kFloat32, device);
//                target_wh[best_anchor.item<int>()][gy][gx] = torch::log(torch::tensor({tw / anchor_boxes[best_anchor][0].item<float>(),
//                                                                                       th / anchor_boxes[best_anchor][1].item<float>()},
//                                                                                      torch::kFloat32, device) + 1e-6);
//                target_conf[best_anchor.item<int>()][gy][gx] = 1.0;
//                target_class[best_anchor.item<int>()][gy][gx][class_id] = 1.0;
//            }
//
//            // Compute losses
//            auto pred_xy = pred_b.slice(4, 0, 2); // [num_anchors, grid_h, grid_w, 2]
//            auto pred_wh = pred_b.slice(4, 2, 4); // [num_anchors, grid_h, grid_w, 2]
//            auto pred_conf = pred_b.select(4, 4); // [num_anchors, grid_h, grid_w]
//            auto pred_class = pred_b.slice(4, 5, 5 + num_classes_); // [num_anchors, grid_h, grid_w, num_classes]
//
//            loss_xy += lambda_coord_ * torch::nn::functional::mse_loss(pred_xy.masked_select(obj_mask.unsqueeze(3)),
//                                                                       target_xy.masked_select(obj_mask.unsqueeze(3)));
//            loss_wh += lambda_coord_ * torch::nn::functional::mse_loss(pred_wh.masked_select(obj_mask.unsqueeze(3)),
//                                                                       target_wh.masked_select(obj_mask.unsqueeze(3)));
//            loss_conf += torch::nn::functional::binary_cross_entropy(pred_conf.masked_select(obj_mask), target_conf.masked_select(obj_mask)) +
//                         lambda_noobj_ * torch::nn::functional::binary_cross_entropy(pred_conf.masked_select(noobj_mask), target_conf.masked_select(noobj_mask));
//            loss_class += torch::nn::functional::binary_cross_entropy(pred_class.masked_select(obj_mask.unsqueeze(3)),
//                                                                      target_class.masked_select(obj_mask.unsqueeze(3)));
//        }
//
//        return (loss_xy + loss_wh + loss_conf + loss_class) / batch_size;
//    }
//
//    int num_classes_, num_anchors_;
//    float lambda_coord_, lambda_noobj_;
//    torch::Tensor anchors_;
//};
//TORCH_MODULE(YOLOLoss);
//
//// Custom Dataset for Images and Bounding Boxes
//struct YOLODataset : torch::data::Dataset<YOLODataset> {
//    YOLODataset(const std::string& img_dir, const std::string& label_dir, int max_objects = 10)
//            : max_objects_(max_objects) {
//        for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//            if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                image_paths_.push_back(entry.path().string());
//                std::string label_path = label_dir + "/" + entry.path().stem().string() + ".txt";
//                label_paths_.push_back(label_path);
//            }
//        }
//    }
//
//    torch::data::Example<> get(size_t index) override {
//        // Load image
//        cv::Mat image = cv::imread(image_paths_[index], cv::IMREAD_GRAYSCALE);
//        if (image.empty()) {
//            throw std::runtime_error("Failed to load image: " + image_paths_[index]);
//        }
//        image.convertTo(image, CV_32F, 1.0 / 255.0);
//        torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols}, torch::kFloat32);
//
//        // Load labels (format: class x_center y_center width height, normalized)
//        torch::Tensor label_tensor = torch::zeros({max_objects_, 5}, torch::kFloat32);
//        label_tensor.fill_(-1); // Invalid objects
//        std::ifstream infile(label_paths_[index]);
//        if (infile.is_open()) {
//            std::string line;
//            int obj_idx = 0;
//            while (std::getline(infile, line) && obj_idx < max_objects_) {
//                std::istringstream iss(line);
//                float cls, x, y, w, h;
//                if (iss >> cls >> x >> y >> w >> h) {
//                    label_tensor[obj_idx] = torch::tensor({x, y, w, h, cls}, torch::kFloat32);
//                    obj_idx++;
//                }
//            }
//            infile.close();
//        }
//
//        return {img_tensor, label_tensor};
//    }
//
//    torch::optional<size_t> size() const override {
//        return image_paths_.size();
//    }
//
//    std::vector<std::string> image_paths_, label_paths_;
//    int max_objects_;
//};
//
//int main() {
//    try {
//        // Set device
//        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//        // Initialize model and loss
//        int num_classes = 2; // Example: 2 classes (e.g., person, car)
//        int num_anchors = 3;
//        YOLOv10 model(num_classes, num_anchors);
//        YOLOLoss loss(num_classes, num_anchors);
//        model->to(device);
//        loss->to(device);
//
//        // Optimizer
//        torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
//
//        // Load dataset
//        auto dataset = YOLODataset("./data/images", "./data/labels", 10)
//                .map(torch::data::transforms::Stack<>());
//        auto data_loader = torch::data::make_data_loader(
//                dataset, torch::data::DataLoaderOptions().batch_size(32).workers(2));
//
//        // Training loop
//        model->train();
//        for (int epoch = 0; epoch < 20; ++epoch) {
//            float total_loss = 0.0;
//            int batch_count = 0;
//
//            for (auto& batch : *data_loader) {
//                auto images = batch.data.to(device);
//                auto labels = batch.target.to(device);
//
//                optimizer.zero_grad();
//                auto output = model->forward(images);
//                auto loss_value = loss->forward(output, labels);
//                loss_value.backward();
//                optimizer.step();
//
//                total_loss += loss_value.item<float>();
//                batch_count++;
//            }
//
//            std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / batch_count << std::endl;
//        }
//
//        // Save model
//        torch::save(model, "yolov10_trained.pt");
//        std::cout << "Model saved as yolov10_trained.pt" << std::endl;
//
//    } catch (const std::exception& e) {
//        std::cerr << "Error: " << e.what() << std::endl;
//        return -1;
//    }
//
//    return 0;
//}

//YOLO1 GROK


// #include <torch/torch.h>
// #include <torch/script.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <fstream>
//
// // YOLOv1 Backbone (Simplified Darknet)
// struct BackboneImpl : torch::nn::Module {
//     BackboneImpl() {
//         conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1)));
//         conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1)));
//         conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(1).padding(1)));
//         conv4 = register_module("conv4", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).stride(2).padding(1)));
//         relu = register_module("relu", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
//         pool = register_module("pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = relu->forward(conv1->forward(x)); // [batch, 16, 28, 28]
//         x = pool->forward(x); // [batch, 16, 14, 14]
//         x = relu->forward(conv2->forward(x)); // [batch, 32, 7, 7]
//         x = relu->forward(conv3->forward(x)); // [batch, 64, 7, 7]
//         x = relu->forward(conv4->forward(x)); // [batch, 128, 3, 3]
//         return x;
//     }
//
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv4{nullptr};
//     torch::nn::LeakyReLU relu{nullptr};
//     torch::nn::MaxPool2d pool{nullptr};
// };
// TORCH_MODULE(Backbone);
//
// // YOLOv1 Head
// struct HeadImpl : torch::nn::Module {
//     HeadImpl(int num_classes, int boxes_per_cell = 2, int grid_size = 7)
//         : num_classes_(num_classes), boxes_per_cell_(boxes_per_cell), grid_size_(grid_size) {
//         fc1 = register_module("fc1", torch::nn::Linear(128 * 3 * 3, 512));
//         fc2 = register_module("fc2", torch::nn::Linear(512, grid_size * grid_size * (boxes_per_cell * 5 + num_classes)));
//         relu = register_module("relu", torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(0.1)));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch_size = x.size(0);
//         x = x.view({batch_size, -1}); // [batch, 128 * 3 * 3]
//         x = relu->forward(fc1->forward(x)); // [batch, 512]
//         x = fc2->forward(x); // [batch, 7 * 7 * (2 * 5 + num_classes)]
//         x = x.view({batch_size, grid_size_, grid_size_, boxes_per_cell_ * 5 + num_classes_});
//         // Apply sigmoid to x, y, conf, and class probs
//         for (int b = 0; b < boxes_per_cell_; ++b) {
//             x.slice(3, b * 5, b * 5 + 2) = torch::sigmoid(x.slice(3, b * 5, b * 5 + 2)); // x, y
//             x.slice(3, b * 5 + 4, b * 5 + 5) = torch::sigmoid(x.slice(3, b * 5 + 4, b * 5 + 5)); // conf
//         }
//         x.slice(3, boxes_per_cell_ * 5, boxes_per_cell_ * 5 + num_classes_) = torch::sigmoid(x.slice(3, boxes_per_cell_ * 5, boxes_per_cell_ * 5 + num_classes_)); // classes
//         return x; // [batch, 7, 7, 2 * 5 + num_classes]
//     }
//
//     int num_classes_, boxes_per_cell_, grid_size_;
//     torch::nn::Linear fc1{nullptr}, fc2{nullptr};
//     torch::nn::LeakyReLU relu{nullptr};
// };
// TORCH_MODULE(Head);
//
// // YOLOv1 Model
// struct YOLOv1Impl : torch::nn::Module {
//     YOLOv1Impl(int num_classes, int boxes_per_cell = 2, int grid_size = 7) {
//         backbone = register_module("backbone", Backbone());
//         head = register_module("head", Head(num_classes, boxes_per_cell, grid_size));
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = backbone->forward(x);
//         x = head->forward(x);
//         return x;
//     }
//
//     Backbone backbone{nullptr};
//     Head head{nullptr};
// };
// TORCH_MODULE(YOLOv1);

// YOLOv1 Loss
// struct YOLOv1LossImpl : torch::nn::Module {
//     YOLOv1LossImpl(int num_classes, int boxes_per_cell = 2, int grid_size = 7, float lambda_coord = 5.0, float lambda_noobj = 0.5)
//         : num_classes_(num_classes), boxes_per_cell_(boxes_per_cell), grid_size_(grid_size),
//           lambda_coord_(lambda_coord), lambda_noobj_(lambda_noobj) {}
//
//     torch::Tensor forward(torch::Tensor pred, torch::Tensor target) {
//         auto batch_size = pred.size(0);
//         auto device = pred.device();
//
//         torch::Tensor loss_xy = torch::zeros({}, device);
//         torch::Tensor loss_wh = torch::zeros({}, device);
//         torch::Tensor loss_conf = torch::zeros({}, device);
//         torch::Tensor loss_class = torch::zeros({}, device);
//
//         for (int b = 0; b < batch_size; ++b) {
//             auto pred_b = pred[b]; // [7, 7, 2 * 5 + num_classes]
//             auto target_b = target[b]; // [max_objects, 5] (x, y, w, h, class)
//
//             auto obj_mask = torch::zeros({grid_size_, grid_size_, boxes_per_cell_}, torch::kBool, device);
//             auto noobj_mask = torch::ones({grid_size_, grid_size_, boxes_per_cell_}, torch::kBool, device);
//             auto target_xy = torch::zeros({grid_size_, grid_size_, boxes_per_cell_, 2}, torch::kFloat32, device);
//             auto target_wh = torch::zeros({grid_size_, grid_size_, boxes_per_cell_, 2}, torch::kFloat32, device);
//             auto target_conf = torch::zeros({grid_size_, grid_size_, boxes_per_cell_}, torch::kFloat32, device);
//             auto target_class = torch::zeros({grid_size_, grid_size_, num_classes_}, torch::kFloat32, device);
//
//             for (int t = 0; t < target_b.size(0); ++t) {
//                 if (target_b[t][4].item<float>() < 0) continue;
//                 float tx = target_b[t][0].item<float>() * grid_size_;
//                 float ty = target_b[t][1].item<float>() * grid_size_;
//                 float tw = std::sqrt(target_b[t][2].item<float>());
//                 float th = std::sqrt(target_b[t][3].item<float>());
//                 int class_id = target_b[t][4].item<float>();
//                 int gx = static_cast<int>(tx);
//                 int gy = static_cast<int>(ty);
//                 if (gx >= grid_size_ || gy >= grid_size_ || gx < 0 || gy < 0) continue;
//
//                 // Assign to best box (simplified: first box)
//                 int best_box = 0; // YOLOv1 assigns one box per object
//                 obj_mask[gy][gx][best_box] = true;
//                 noobj_mask[gy][gx][best_box] = false;
//                 target_xy[gy][gx][best_box] = torch::tensor({tx - gx, ty - gy}, torch::kFloat32, device);
//                 target_wh[gy][gx][best_box] = torch::tensor({tw, th}, torch::kFloat32, device);
//                 target_conf[gy][gx][best_box] = 1.0;
//                 target_class[gy][gx][class_id] = 1.0;
//             }
//
//             for (int i = 0; i < grid_size_; ++i) {
//                 for (int j = 0; j < grid_size_; ++j) {
//                     for (int k = 0; k < boxes_per_cell_; ++k) {
//                         auto pred_xy_k = pred_b[i][j].slice(0, k * 5, k * 5 + 2);
//                         auto pred_wh_k = pred_b[i][j].slice(0, k * 5 + 2, k * 5 + 4);
//                         auto pred_conf_k = pred_b[i][j][k * 5 + 4];
//                         if (obj_mask[i][j][k]) {
//                             loss_xy += lambda_coord_ * torch::nn::functional::mse_loss(pred_xy_k, target_xy[i][j][k]);
//                             loss_wh += lambda_coord_ * torch::nn::functional::mse_loss(pred_wh_k, target_wh[i][j][k]);
//                             loss_conf += torch::nn::functional::mse_loss(pred_conf_k, target_conf[i][j][k]);
//                         } else {
//                             loss_conf += lambda_noobj_ * torch::nn::functional::mse_loss(pred_conf_k, target_conf[i][j][k]);
//                         }
//                     }
//                     if (obj_mask[i][j].any().item<bool>()) {
//                         auto pred_class = pred_b[i][j].slice(0, boxes_per_cell_ * 5, boxes_per_cell_ * 5 + num_classes_);
//                         loss_class += torch::nn::functional::mse_loss(pred_class, target_class[i][j]);
//                     }
//                 }
//             }
//         }
//
//         return (loss_xy + loss_wh + loss_conf + loss_class) / batch_size;
//     }
//
//     int num_classes_, boxes_per_cell_, grid_size_;
//     float lambda_coord_, lambda_noobj_;
// };
// TORCH_MODULE(YOLOv1Loss);
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
// bool LoadPretrainedWeights(YOLOv1& model, const std::string& pretrained_path, torch::Device device) {
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
//         int boxes_per_cell = 2;
//         int grid_size = 7;
//         YOLOv1 model(num_classes, boxes_per_cell, grid_size);
//         YOLOv1Loss loss(num_classes, boxes_per_cell, grid_size);
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
//         torch::save(model, "yolov1_finetuned.pt");
//         std::cout << "Fine-tuned model saved as yolov1_finetuned.pt" << std::endl;
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
    Backbone::Backbone()
    {
        conv1 = register_module(
            "conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, 3).stride(1).padding(1)));
        conv2 = register_module(
            "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, 3).stride(2).padding(1)));
        conv3 = register_module(
            "conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).stride(2).padding(1)));
        relu = register_module("relu", torch::nn::ReLU());
    }

    auto Backbone::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }


    torch::Tensor Backbone::forward(torch::Tensor x)
    {
        x = relu->forward(conv1->forward(x)); // [batch, 16, 28, 28]
        x = relu->forward(conv2->forward(x)); // [batch, 32, 14, 14]
        x = relu->forward(conv3->forward(x)); // [batch, 64, 7, 7]
        return x;
    }

    Neck::Neck()
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 32, 1).stride(1)));
        upsample = register_module(
            "upsample", torch::nn::Upsample(
                torch::nn::UpsampleOptions().scale_factor(vector<double>{2.0}).mode(torch::kNearest)));
        conv2 = register_module(
            "conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 16, 3).stride(1).padding(1)));
        relu = register_module("relu", torch::nn::ReLU());
    }

    auto Neck::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }


    torch::Tensor Neck::forward(torch::Tensor x)
    {
        x = relu->forward(conv1->forward(x)); // [batch, 32, 7, 7]
        x = upsample->forward(x); // [batch, 32, 14, 14]
        x = relu->forward(conv2->forward(x)); // [batch, 16, 14, 14]
        return x;
    }


    Head::Head(int num_classes, int num_anchors) : num_classes_(num_classes), num_anchors_(num_anchors)
    {
        // Output: [x, y, w, h, conf, classes] per anchor
        conv = register_module(
            "conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, num_anchors * (5 + num_classes), 1).stride(1)));
    }

    auto Head::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }


    torch::Tensor Head::forward(torch::Tensor x)
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


    YOLOv10::YOLOv10(int num_classes, int num_anchors)
    {
        backbone = register_module("backbone", std::make_shared<Backbone>());
        neck = register_module("neck", std::make_shared<Neck>());
        head = register_module("head", std::make_shared<Head>(num_classes, num_anchors));
    }

    auto YOLOv10::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];
        return this->forward(x);
    }


    torch::Tensor YOLOv10::forward(torch::Tensor x)
    {
        x = backbone->forward(x);
        x = neck->forward(x);
        x = head->forward(x);
        return x;
    }

    // YOLOLoss::YOLOLoss(int num_classes, int num_anchors, float lambda_coord = 5.0, float lambda_noobj = 0.5)
    //     : num_classes_(num_classes), num_anchors_(num_anchors), lambda_coord_(lambda_coord),
    //       lambda_noobj_(lambda_noobj)
    // {
    //     // Predefined anchors (width, height) scaled to grid
    //     anchors_ = torch::tensor({{1.0, 1.0}, {2.0, 2.0}, {0.5, 0.5}}, torch::kFloat32);
    // }
    //
    // torch::Tensor YOLOLoss::forward(torch::Tensor pred, torch::Tensor target)
    // {
    //     auto batch_size = pred.size(0);
    //     auto grid_size = pred.size(2); // 14x14
    //     auto device = pred.device();
    //
    //     // Initialize losses
    //     torch::Tensor loss_xy = torch::zeros({}, device);
    //     torch::Tensor loss_wh = torch::zeros({}, device);
    //     torch::Tensor loss_conf = torch::zeros({}, device);
    //     torch::Tensor loss_class = torch::zeros({}, device);
    //
    //     // Generate grid offsets
    //     auto grid_x = torch::arange(grid_size, torch::kFloat32, device).repeat({grid_size, 1});
    //     auto grid_y = grid_x.transpose(0, 1);
    //     grid_x = grid_x.view({1, 1, grid_size, grid_size, 1});
    //     grid_y = grid_y.view({1, 1, grid_size, grid_size, 1});
    //
    //     // Process each batch
    //     for (int b = 0; b < batch_size; ++b)
    //     {
    //         auto pred_b = pred[b]; // [num_anchors, grid_h, grid_w, 5 + num_classes]
    //         auto target_b = target[b]; // [max_objects, 5] (x, y, w, h, class)
    //
    //         // Create target tensors
    //         auto obj_mask = torch::zeros({num_anchors_, grid_size, grid_size}, torch::kBool, device);
    //         auto noobj_mask = torch::ones({num_anchors_, grid_size, grid_size}, torch::kBool, device);
    //         auto target_xy = torch::zeros({num_anchors_, grid_size, grid_size, 2}, torch::kFloat32, device);
    //         auto target_wh = torch::zeros({num_anchors_, grid_size, grid_size, 2}, torch::kFloat32, device);
    //         auto target_conf = torch::zeros({num_anchors_, grid_size, grid_size}, torch::kFloat32, device);
    //         auto target_class = torch::zeros({num_anchors_, grid_size, grid_size, num_classes_}, torch::kFloat32,
    //                                          device);
    //
    //         // Assign ground truth to grid cells
    //         for (int t = 0; t < target_b.size(0); ++t)
    //         {
    //             if (target_b[t][4].item<float>() < 0) continue; // Invalid object
    //             float tx = target_b[t][0].item<float>() * grid_size;
    //             float ty = target_b[t][1].item<float>() * grid_size;
    //             float tw = target_b[t][2].item<float>() * grid_size;
    //             float th = target_b[t][3].item<float>() * grid_size;
    //             int class_id = target_b[t][4].item<float>();
    //             int gx = static_cast<int>(tx);
    //             int gy = static_cast<int>(ty);
    //             if (gx >= grid_size || gy >= grid_size || gx < 0 || gy < 0) continue;
    //
    //             // Compute IoU with anchors to select best anchor
    //             auto anchor_boxes = anchors_.clone().to(device);
    //             auto gt_box = torch::tensor({tw, th}, torch::kFloat32, device).unsqueeze(0);
    //             auto anchor_w = anchor_boxes.select(1, 0);
    //             auto anchor_h = anchor_boxes.select(1, 1);
    //             auto inter_w = torch::min(anchor_w, gt_box[0][0]);
    //             auto inter_h = torch::min(anchor_h, gt_box[0][1]);
    //             auto inter_area = inter_w * inter_h;
    //             auto union_area = anchor_w * anchor_h + gt_box[0][0] * gt_box[0][1] - inter_area;
    //             auto iou = inter_area / (union_area + 1e-6);
    //             auto [_, best_anchor] = torch::max(iou, 0);
    //
    //             // Assign target
    //             obj_mask[best_anchor.item<int>()][gy][gx] = true;
    //             noobj_mask[best_anchor.item<int>()][gy][gx] = false;
    //             target_xy[best_anchor.item<int>()][gy][gx] = torch::tensor(
    //                 {tx - gx, ty - gy}, torch::kFloat32, device);
    //             target_wh[best_anchor.item<int>()][gy][gx] = torch::log(torch::tensor({
    //                     tw / anchor_boxes[best_anchor][0].item<float>(),
    //                     th / anchor_boxes[best_anchor][1].item<float>()
    //                 },
    //                 torch::kFloat32, device) + 1e-6);
    //             target_conf[best_anchor.item<int>()][gy][gx] = 1.0;
    //             target_class[best_anchor.item<int>()][gy][gx][class_id] = 1.0;
    //         }
    //
    //         // Compute losses
    //         auto pred_xy = pred_b.slice(4, 0, 2); // [num_anchors, grid_h, grid_w, 2]
    //         auto pred_wh = pred_b.slice(4, 2, 4); // [num_anchors, grid_h, grid_w, 2]
    //         auto pred_conf = pred_b.select(4, 4); // [num_anchors, grid_h, grid_w]
    //         auto pred_class = pred_b.slice(4, 5, 5 + num_classes_); // [num_anchors, grid_h, grid_w, num_classes]
    //
    //         loss_xy += lambda_coord_ * torch::nn::functional::mse_loss(pred_xy.masked_select(obj_mask.unsqueeze(3)),
    //                                                                    target_xy.masked_select(
    //                                                                        obj_mask.unsqueeze(3)));
    //         loss_wh += lambda_coord_ * torch::nn::functional::mse_loss(pred_wh.masked_select(obj_mask.unsqueeze(3)),
    //                                                                    target_wh.masked_select(
    //                                                                        obj_mask.unsqueeze(3)));
    //         loss_conf += torch::nn::functional::binary_cross_entropy(
    //                 pred_conf.masked_select(obj_mask), target_conf.masked_select(obj_mask)) +
    //             lambda_noobj_ * torch::nn::functional::binary_cross_entropy(
    //                 pred_conf.masked_select(noobj_mask), target_conf.masked_select(noobj_mask));
    //         loss_class += torch::nn::functional::binary_cross_entropy(
    //             pred_class.masked_select(obj_mask.unsqueeze(3)),
    //             target_class.masked_select(obj_mask.unsqueeze(3)));
    //     }
    //
    //     return (loss_xy + loss_wh + loss_conf + loss_class) / batch_size;
    // }
}
