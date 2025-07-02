#include "include/models/others/deep_lab_cut.h"


using namespace std;

//
// #include <torch/torch.h>
// #include <torchvision/vision.h>
// #include <iostream>
// #include <vector>
// #include <fstream>
// #include <string>
//
// // Custom Dataset for Pose Estimation
// struct PoseDataset : torch::data::Dataset<PoseDataset> {
//     std::vector<std::string> image_paths;
//     std::vector<torch::Tensor> heatmaps;
//     int img_size = 224;
//     float sigma = 3.0; // For Gaussian heatmap
//
//     PoseDataset(const std::string& csv_file, const std::string& image_dir) {
//         // Load CSV file with image paths and keypoints
//         std::ifstream file(csv_file);
//         std::string line;
//         std::getline(file, line); // Skip header
//         while (std::getline(file, line)) {
//             std::vector<std::string> tokens;
//             std::stringstream ss(line);
//             std::string token;
//             while (std::getline(ss, token, ',')) {
//                 tokens.push_back(token);
//             }
//             image_paths.push_back(image_dir + "/" + tokens[0]);
//             std::vector<float> keypoints;
//             for (size_t i = 1; i < tokens.size(); i += 2) {
//                 keypoints.push_back(std::stof(tokens[i])); // x
//                 keypoints.push_back(std::stof(tokens[i + 1])); // y
//             }
//             // Generate heatmap for keypoints
//             auto heatmap = generate_heatmap(keypoints);
//             heatmaps.push_back(heatmap);
//         }
//     }
//
//     torch::Tensor generate_heatmap(const std::vector<float>& keypoints) {
//         auto heatmap = torch::zeros({static_cast<long>(keypoints.size() / 2), img_size, img_size});
//         for (size_t i = 0; i < keypoints.size(); i += 2) {
//             int x = static_cast<int>(keypoints[i]);
//             int y = static_cast<int>(keypoints[i + 1]);
//             if (x >= 0 && x < img_size && y >= 0 && y < img_size) {
//                 for (int h = 0; h < img_size; ++h) {
//                     for (int w = 0; w < img_size; ++w) {
//                         float val = std::exp(-((w - x) * (w - x) + (h - y) * (h - y)) / (2 * sigma * sigma));
//                         heatmap[i / 2][h][w] = val;
//                     }
//                 }
//             }
//         }
//         return heatmap;
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         // Load image (simplified, assumes preprocessed images)
//         auto image = torch::rand({3, img_size, img_size}); // Placeholder: replace with actual image loading
//         auto heatmap = heatmaps[index];
//         return {image, heatmap};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths.size();
//     }
// };
//
// // DeepLabCut-inspired Model
// struct DLCModelImpl : torch::nn::Module {
//     torch::nn::Sequential backbone{nullptr};
//     torch::nn::ConvTranspose2d deconv1{nullptr}, deconv2{nullptr};
//     int num_keypoints;
//
//     DLCModelImpl(int num_keypoints_) : num_keypoints(num_keypoints_) {
//         // Initialize ResNet-50 backbone
//         auto resnet = torchvision::models::ResNet50();
//         backbone = register_module("backbone", resnet->features);
//         deconv1 = register_module(
//             "deconv1",
//             torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(2048, 256, 4).stride(2).padding(1))
//         );
//         deconv2 = register_module(
//             "deconv2",
//             torch::nn::ConvTranspose2d(torch::nn::ConvTranspose2dOptions(256, num_keypoints, 4).stride(2).padding(1))
//         );
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         x = backbone->forward(x); // [batch, 2048, H/32, W/32]
//         x = torch::relu(deconv1->forward(x)); // Upsample
//         x = deconv2->forward(x); // [batch, num_keypoints, H, W]
//         return x;
//     }
// };
// TORCH_MODULE(DLCModel);
//
// // Main training function
// int main() {
//     // Device configuration
//     torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//     std::cout << "Training on: " << (device.type() == torch::kCUDA ? "GPU" : "CPU") << std::endl;
//
//     // Hyperparameters
//     const int batch_size = 16;
//     const int num_epochs = 100;
//     const float learning_rate = 0.001;
//     const int num_keypoints = 5; // Number of keypoints to track
//
//     // Initialize model and optimizer
//     auto model = DLCModel(num_keypoints);
//     model->to(device);
//     auto optimizer = torch::optim::Adam(model->parameters(), learning_rate);
//
//     // Load dataset
//     auto dataset = PoseDataset("keypoints.csv", "images")
//         .map(torch::data::transforms::Stack<>());
//     auto data_loader = torch::data::make_data_loader(
//         dataset,
//         torch::data::DataLoaderOptions().batch_size(batch_size).workers(4)
//     );
//
//     // Training loop
//     model->train();
//     for (int epoch = 0; epoch < num_epochs; ++epoch) {
//         float total_loss = 0.0;
//         for (auto& batch : *data_loader) {
//             auto images = batch.data.to(device);
//             auto heatmaps = batch.target.to(device);
//
//             optimizer.zero_grad();
//             auto output = model->forward(images);
//             auto loss = torch::mse_loss(output, heatmaps);
//             loss.backward();
//             optimizer.step();
//
//             total_loss += loss.item<float>() * images.size(0);
//         }
//
//         std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
//                   << "], Loss: " << total_loss / dataset.size().value() << std::endl;
//     }
//
//     // Save model
//     torch::save(model, "dlc_model.pt");
//     std::cout << "Model saved to dlc_model.pt" << std::endl;
//
//     return 0;
// }

namespace xt::models
{
    DeepLabCut::DeepLabCut(int num_classes, int in_channels)
    {
    }

    DeepLabCut::DeepLabCut(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void DeepLabCut::reset()
    {
    }

    auto DeepLabCut::forward(std::initializer_list<std::any> tensors) -> std::any
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
