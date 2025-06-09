#include "include/models/others/nerf.h"


using namespace std;


#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <random>

// Positional Encoding
torch::Tensor positional_encoding(const torch::Tensor& x, int L) {
    std::vector<torch::Tensor> encodings;
    encodings.push_back(x);
    for (int i = 0; i < L; ++i) {
        float freq = std::pow(2.0f, static_cast<float>(i));
        encodings.push_back(torch::sin(freq * x));
        encodings.push_back(torch::cos(freq * x));
    }
    return torch::cat(encodings, -1);
}

// NeRF MLP Model
struct NeRFImpl : torch::nn::Module {
    torch::nn::Sequential xyz_layers{nullptr}, view_layers{nullptr};
    int L_xyz = 10, L_dir = 4;

    NeRFImpl() {
        // Layers for 3D coordinates (63 = 3 + 2*3*L_xyz)
        xyz_layers = register_module("xyz_layers", torch::nn::Sequential(
            torch::nn::Linear(3 + 2 * 3 * L_xyz, 256),
            torch::nn::ReLU(true),
            torch::nn::Linear(256, 256),
            torch::nn::ReLU(true),
            torch::nn::Linear(256, 256),
            torch::nn::ReLU(true),
            torch::nn::Linear(256, 256),
            torch::nn::ReLU(true),
            torch::nn::Linear(256, 256 + 1) // Outputs density + feature
        ));
        // Layers for view direction (27 = 3 + 2*3*L_dir)
        view_layers = register_module("view_layers", torch::nn::Sequential(
            torch::nn::Linear(256 + 3 + 2 * 3 * L_dir, 128),
            torch::nn::ReLU(true),
            torch::nn::Linear(128, 3), // RGB output
            torch::nn::Sigmoid() // Ensure RGB in [0,1]
        ));
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& xyz, const torch::Tensor& dir) {
        auto xyz_encoded = positional_encoding(xyz, L_xyz); // [batch, num_samples, 63]
        auto xy_out = xyz_layers->forward(xyz_encoded); // [batch, num_samples, 256+1]
        auto sigma = torch::relu(xy_out.slice(-1, 256, 257)); // Density
        auto feature = xy_out.slice(-1, 0, 256); // Feature
        auto dir_encoded = positional_encoding(dir, L_dir); // [batch, num_samples, 27]
        auto input = torch::cat({feature, dir_encoded}, -1); // [batch, num_samples, 256+27]
        auto rgb = view_layers->forward(input); // [batch, num_samples, 3]
        return {rgb, sigma};
    }
};
TORCH_MODULE(NeRF);

// Custom Dataset for NeRF
struct NeRFDataset : torch::data::Dataset<NeRFDataset> {
    std::vector<std::string> image_paths;
    std::vector<torch::Tensor> poses;
    torch::Tensor focal;
    int img_height = 800, img_width = 800;

    NeRFDataset(const std::string& data_dir) {
        // Load poses and image paths
        std::ifstream file(data_dir + "/poses.txt");
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string img_path;
            std::vector<float> pose_data(12); // 3x4 matrix
            ss >> img_path;
            for (int i = 0; i < 12; ++i) {
                ss >> pose_data[i];
            }
            image_paths.push_back(data_dir + "/" + img_path);
            auto pose = torch::from_blob(pose_data.data(), {3, 4}).clone();
            poses.push_back(pose);
        }
        focal = torch::tensor(1111.111); // Example focal length (from NeRF dataset)
    }

    torch::data::Example<> get(size_t index) override {
        // Load image
        cv::Mat img = cv::imread(image_paths[index], cv::IMREAD_COLOR);
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        cv::resize(img, img, cv::Size(img_width, img_height));
        auto img_tensor = torch::from_blob(img.data, {img_height, img_width, 3}, torch::kByte)
            .to(torch::kFloat) / 255.0;
        img_tensor = img_tensor.permute({2, 0, 1}); // [3, H, W]

        // Get pose
        auto pose = poses[index];

        return {img_tensor, pose};
    }

    torch::optional<size_t> size() const override {
        return image_paths.size();
    }
};

// Ray generation and volumetric rendering
struct RaySampler {
    static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_rays(
        int H, int W, const torch::Tensor& pose, const torch::Tensor& focal
    ) {
        auto device = pose.device();
        auto i = torch::arange(W, torch::device(device).dtype(torch::kFloat));
        auto j = torch::arange(H, torch::device(device).dtype(torch::kFloat));
        auto [i_grid, j_grid] = torch::meshgrid({i, j}, "ij");
        auto dirs = torch::stack({
            (i_grid - W * 0.5) / focal,
            -(j_grid - H * 0.5) / focal,
            -torch::ones_like(i_grid)
        }, -1); // [H, W, 3]
        auto rays_d = torch::matmul(pose.slice(1, 0, 3), dirs.view(-1, 3).t()).t(); // [H*W, 3]
        auto rays_o = pose.slice(1, 3, 4).repeat({H * W, 1}); // [H*W, 3]
        return {rays_o, rays_d, dirs.view(-1, 3)};
    }

    static torch::Tensor render_rays(
        const torch::Tensor& rgb, const torch::Tensor& sigma,
        const torch::Tensor& t_vals, const torch::Tensor& rays_d
    ) {
        auto delta = t_vals.slice(1, 1) - t_vals.slice(1, 0, -1); // [batch, N-1]
        auto dists = torch::cat({delta, torch::ones({delta.size(0), 1}, delta.options()) * 1e10}, -1); // [batch, N]
        dists = dists * torch::norm(rays_d, 2, -1, true); // Adjust for ray direction norm
        auto alpha = 1.0 - torch::exp(-sigma * dists); // [batch, N, 1]
        auto T = torch::cumprod(1.0 - alpha + 1e-10, 1); // [batch, N, 1]
        T = torch::cat({torch::ones({T.size(0), 1, 1}, T.options()), T.slice(1, 0, -1)}, 1); // [batch, N, 1]
        auto weights = alpha * T; // [batch, N, 1]
        auto rgb_map = (weights * rgb).sum(1); // [batch, 3]
        return rgb_map;
    }
};

// Main training function
int main() {
    // Device configuration
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Training on: " << (device.type() == torch::kCUDA ? "GPU" : "CPU") << std::endl;

    // Hyperparameters
    const int batch_size = 1024; // Number of rays
    const int num_epochs = 200;
    const float learning_rate = 5e-4;
    const int num_samples = 64; // Samples per ray
    const float near = 2.0, far = 6.0; // Depth bounds

    // Initialize model and optimizer
    auto model = NeRF();
    model->to(device);
    auto optimizer = torch::optim::Adam(model->parameters(), learning_rate);

    // Load dataset
    auto dataset = NeRFDataset("data/lego")
        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(1).workers(4)
    );

    // Random number generator for ray sampling
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, 800 * 800 - 1);

    // Training loop
    model->train();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0;
        int batch_count = 0;
        for (auto& batch : *data_loader) {
            auto img = batch.data.to(device); // [3, H, W]
            auto pose = batch.target.to(device); // [3, 4]
            auto [rays_o, rays_d, view_dirs] = RaySampler::get_rays(800, 800, pose, dataset.focal.to(device));

            // Sample random rays
            std::vector<int64_t> idx(batch_size);
            for (int i = 0; i < batch_size; ++i) {
                idx[i] = dist(gen);
            }
            auto rays_o_batch = rays_o.index({torch::tensor(idx, torch::kLong).to(device)}); // [batch, 3]
            auto rays_d_batch = rays_d.index({torch::tensor(idx, torch::kLong).to(device)});
            auto target_rgb = img.view({3, -1}).t().index({torch::tensor(idx, torch::kLong).to(device)}); // [batch, 3]

            // Sample points along rays
            auto t_vals = torch::linspace(near, far, num_samples, torch::device(device));
            auto points = rays_o_batch.unsqueeze(1) + rays_d_batch.unsqueeze(1) * t_vals.unsqueeze(0).unsqueeze(-1); // [batch, N, 3]
            points = points.view({batch_size * num_samples, 3});
            view_dirs = view_dirs.index({torch::tensor(idx, torch::kLong).to(device)});
            view_dirs = view_dirs.unsqueeze(1).repeat({1, num_samples, 1}).view({batch_size * num_samples, 3});

            optimizer.zero_grad();
            auto [rgb, sigma] = model->forward(points, view_dirs);
            rgb = rgb.view({batch_size, num_samples, 3});
            sigma = sigma.view({batch_size, num_samples, 1});
            auto rgb_map = RaySampler::render_rays(rgb, sigma, t_vals, rays_d_batch);
            auto loss = torch::mse_loss(rgb_map, target_rgb);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            batch_count++;
        }

        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                  << "], Loss: " << total_loss / batch_count << std::endl;
    }

    // Save model
    torch::save(model, "nerf.pt");
    std::cout << "Model saved to nerf.pt" << std::endl;

    return 0;
}



namespace xt::models
{
    NeRF::NeRF(int num_classes, int in_channels)
    {
    }

    NeRF::NeRF(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void NeRF::reset()
    {
    }

    auto NeRF::forward(std::initializer_list<std::any> tensors) -> std::any
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
