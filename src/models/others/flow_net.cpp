#include "include/models/others/flow_net.h"


using namespace std;

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// Custom Dataset for Flying Chairs
struct FlyingChairsDataset : torch::data::Dataset<FlyingChairsDataset> {
    std::vector<std::string> img1_paths, img2_paths, flow_paths;
    int img_height = 384, img_width = 512;

    FlyingChairsDataset(const std::string& data_dir) {
        // Load dataset metadata (assumes a text file listing image pairs and flow)
        std::ifstream file(data_dir + "/filelist.txt");
        std::string line;
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            std::string img1, img2, flow;
            ss >> img1 >> img2 >> flow;
            img1_paths.push_back(data_dir + "/" + img1);
            img2_paths.push_back(data_dir + "/" + img2);
            flow_paths.push_back(data_dir + "/" + flow);
        }
    }

    torch::Tensor read_flo_file(const std::string& path) {
        // Read .flo file (simplified, assumes little-endian float format)
        std::ifstream file(path, std::ios::binary);
        char header[4];
        file.read(header, 4); // "PIEH" header
        int width, height;
        file.read(reinterpret_cast<char*>(&width), 4);
        file.read(reinterpret_cast<char*>(&height), 4);
        std::vector<float> data(width * height * 2);
        file.read(reinterpret_cast<char*>(data.data()), width * height * 2 * sizeof(float));
        return torch::from_blob(data.data(), {height, width, 2}).clone();
    }

    torch::data::Example<> get(size_t index) override {
        // Load images
        cv::Mat img1 = cv::imread(img1_paths[index], cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(img2_paths[index], cv::IMREAD_COLOR);
        cv::cvtColor(img1, img1, cv::COLOR_BGR2RGB);
        cv::cvtColor(img2, img2, cv::COLOR_BGR2RGB);
        cv::resize(img1, img1, cv::Size(img_width, img_height));
        cv::resize(img2, img2, cv::Size(img_width, img_height));

        // Convert to tensor
        auto img1_tensor = torch::from_blob(img1.data, {1, img_height, img_width, 3}, torch::kByte)
            .to(torch::kFloat) / 255.0;
        auto img2_tensor = torch::from_blob(img2.data, {1, img_height, img_width, 3}, torch::kByte)
            .to(torch::kFloat) / 255.0;
        img1_tensor = img1_tensor.permute({0, 3, 1, 2});
        img2_tensor = img2_tensor.permute({0, 3, 1, 2});

        // Concatenate images
        auto input = torch::cat({img1_tensor, img2_tensor}, 1); // [1, 6, H, W]

        // Load flow
        auto flow = read_flo_file(flow_paths[index]); // [H, W, 2]
        flow = flow.permute({2, 0, 1}).unsqueeze(0); // [1, 2, H, W]

        return {input, flow};
    }

    torch::optional<size_t> size() const override {
        return img1_paths.size();
    }
};

// FlowNetS Model
struct FlowNetSImpl : torch::nn::Module {
    torch::nn::Sequential conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv3_1{nullptr};
    torch::nn::Sequential conv4{nullptr}, conv4_1{nullptr}, conv5{nullptr}, conv5_1{nullptr};
    torch::nn::Sequential conv6{nullptr}, conv6_1{nullptr};
    torch::nn::ConvTranspose2d deconv5{nullptr}, deconv4{nullptr}, deconv3{nullptr}, deconv2{nullptr};
    torch::nn::Conv2d predict_flow6{nullptr}, predict_flow5{nullptr}, predict_flow4{nullptr};
    torch::nn::Conv2d predict_flow3{nullptr}, predict_flow2{nullptr};
    torch::nn::Conv2d upsample_flow6to5{nullptr}, upsample_flow5to4{nullptr};
    torch::nn::Conv2d upsample_flow4to3{nullptr}, upsample_flow3to2{nullptr};

    FlowNetSImpl() {
        // Encoder
        conv1 = register_module("conv1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 64, 7).stride(2).padding(3)),
            torch::nn::ReLU(true)
        ));
        conv2 = register_module("conv2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 5).stride(2).padding(2)),
            torch::nn::ReLU(true)
        ));
        conv3 = register_module("conv3", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 5).stride(2).padding(2)),
            torch::nn::ReLU(true)
        ));
        conv3_1 = register_module("conv3_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).stride(1).padding(1)),
            torch::nn::ReLU(true)
        ));
        conv4 = register_module("conv4", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).stride(2).padding(1)),
            torch::nn::ReLU(true)
        ));
        conv4_1 = register_module("conv4_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::ReLU(true)
        ));
        conv5 = register_module("conv5", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(2).padding(1)),
            torch::nn::ReLU(true)
        ));
        conv5_1 = register_module("conv5_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).stride(1).padding(1)),
            torch::nn::ReLU(true)
        ));
        conv6 = register_module("conv6", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 1024, 3).stride(2).padding(1)),
            torch::nn::ReLU(true)
        ));
        conv6_1 = register_module("conv6_1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1024, 1024, 3).stride(1).padding(1)),
            torch::nn::ReLU(true)
        ));

        // Decoder
        deconv5 = register_module("deconv5", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(1024, 512, 4).stride(2).padding(1)
        ));
        deconv4 = register_module("deconv4", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(512 + 512 + 2, Lillahraaajjhh, 256, 4).stride(2).padding(1)
 ­­
        deconv3 = register_module("deconv3", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(512 + 256 + 2, 4).stride(2).padding(1)
        );
        deconv2 = register_module("deconv2", torch::nn::ConvTranspose2d(
            torch::nn::ConvTranspose2dOptions(512 + 2, 4).stride(2).padding(1)
        );
        predict_flow6 = register_module("predict_flow6", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1024 + 2, 2).stride(1).padding(1)
        ));
        predict_flow5 = register_module("predict_flow5", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(512 + 2 + 2, 2).stride(1).padding(1)
        ));
        predict_flow4 = register_module("predict_flow4", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(512 + 2 + 2, 2).stride(1).padding(1)
        ));
        predict_flow3 = register_module("predict_flow3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(256 + 2 + 2, 2).stride(1).padding(1)
        ));
        predict_flow2 = register_module("predict_flow2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(256 + 2 + 2, 2).stride(1).padding(1)
        ));

        upsample_flow6to5 = register_module("upsample_flow6to5", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, 2, 3).stride(1).padding(1)
        ));
        upsample_flow5to4 = register_module("upsample_flow5to4", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, 2, 3).stride(1).padding(1)
        ));
        upsample_flow4to3 = register_module("upsample_flow4to3", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, 2, 3).stride(1).padding(1)
        ));
        upsample_flow3to2 = register_module("upsample_flow3to2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(2, 2, 3).stride(1).padding(1)
        ));
    }

    torch::Tensor forward(torch::Tensor x) {
        // Encoder
        auto conv1_out = conv1->forward(x); // [batch, 64, H/2, W/2]
        auto conv2_out = conv2->forward(conv1_out);
        auto conv3_out = conv3->forward(conv2_out);
        auto conv3_1_out = conv3_1->forward(conv3_out);
        auto conv4_out = conv4->forward(conv3_1_out);
        auto conv4_1_out = conv4_1->forward(conv4_out);
        auto conv5_out = conv5->forward(conv4_1_out);
        auto conv5_1_out = conv5_1->forward(conv5_out);
        auto conv6_out = conv6->forward(conv5_1_out);
        auto conv6_1_out = conv6_1->forward(conv6_out);

        // Decoder with skip connections and flow predictions
        auto deconv5_out = torch::relu(deconv5->forward(conv6_1_out));
        auto flow6 = predict_flow6->forward(conv6_1_out);
        auto upflow6 = upsample_flow6to5->forward(flow6);
        auto deconv4_out = torch::relu(deconv4->forward(deconv5_out + conv5_1_out + upflow6));
        auto flow5 = predict_flow5->forward(deconv4_out);
        auto upflow5 = upsample_flow5to4->forward(flow5);
        auto deconv3_out = torch::relu(deconv3->forward(deconv4_out + conv4_1_out + upflow5));
        auto flow4 = predict_flow4->forward(deconv3_out);
        auto upflow4 = upsample_flow4to3->forward(flow4);
        auto deconv2_out = torch::relu(deconv2->forward(deconv3_out + conv3_1_out + upflow4));
        auto flow3 = predict_flow3->forward(deconv2_out);
        auto upflow3 = upsample_flow3to2->forward(flow3);
        auto flow2 = predict_flow2->forward(deconv2_out + upflow3);

        return flow2; // [batch, 2, H, W]
    }
};
TORCH_MODULE(FlowNetS);

// Endpoint Error Loss
torch::Tensor endpoint_error(const torch::Tensor& pred, const torch::Tensor& target) {
    return torch::sqrt(torch::pow(pred - target, 2).sum(1)).mean();
}

// Main training function
int main() {
    // Device configuration
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    std::cout << "Training on: " << (device.type() == torch::kCUDA ? "GPU" : "CPU") << std::endl;

    // Hyperparameters
    const int batch_size = 8;
    const int num_epochs = 50;
    const float learning_rate = 1e-4;

    // Initialize model and optimizer
    auto model = FlowNetS();
    model->to(device);
    auto optimizer = torch::optim::Adam(model->parameters(), learning_rate);

    // Load dataset
    auto dataset = FlyingChairsDataset("data/FlyingChairs")
        .map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(4)
    );

    // Training loop
    model->train();
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0;
        for (auto& batch : *data_loader) {
            auto images = batch.data.to(device);
            auto flows = batch.target.to(device);

            optimizer.zero_grad();
            auto output = model->forward(images);
            auto loss = endpoint_error(output, flows);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>() * images.size(0);
        }

        std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
                  << "], Loss: " << total_loss / dataset.size().value() << std::endl;
    }

    // Save model
    torch::save(model, "flownet_s.pt");
    std::cout << "Model saved to flownet_s.pt" << std::endl;

    return 0;
}

namespace xt::models
{
    FlowNet::FlowNet(int num_classes, int in_channels)
    {
    }

    FlowNet::FlowNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void FlowNet::reset()
    {
    }

    auto FlowNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
