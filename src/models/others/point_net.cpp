#include <models/others/point_net.h>


using namespace std;



// #include <torch/torch.h>
// #include <iostream>
// #include <fstream>
// #include <vector>
// #include <string>
// #include <random>
//
// // T-Net for input/feature transformation
// struct TNetImpl : torch::nn::Module {
//     torch::nn::Sequential layers{nullptr};
//
//     TNetImpl(int in_dim, int out_dim) {
//         layers = register_module("layers", torch::nn::Sequential(
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(in_dim, 64, 1)),
//             torch::nn::BatchNorm1d(64),
//             torch::nn::ReLU(true),
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 1)),
//             torch::nn::BatchNorm1d(128),
//             torch::nn::ReLU(true),
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 1024, 1)),
//             torch::nn::BatchNorm1d(1024),
//             torch::nn::ReLU(true),
//             torch::nn::Functional([](const torch::Tensor& x) { return torch::max_pool1d(x, x.size(2)).squeeze(2); }),
//             torch::nn::Linear(1024, 512),
//             torch::nn::BatchNorm1d(512),
//             torch::nn::ReLU(true),
//             torch::nn::Linear(512, 256),
//             torch::nn::BatchNorm1d(256),
//             torch::nn::ReLU(true),
//             torch::nn::Linear(256, out_dim * out_dim)
//         ));
//
//         // Initialize output as identity matrix
//         auto weight = torch::eye(out_dim).view(-1);
//         layers->ptr()->as<torch::nn::Linear>()->weight = weight;
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         auto batch_size = x.size(0);
//         auto out = layers->forward(x); // [batch, out_dim * out_dim]
//         auto matrix = out.view({batch_size, -1, x.size(1)}) + torch::eye(x.size(1), x.options()).unsqueeze(0); // Add identity
//         return matrix; // [batch, out_dim, in_dim]
//     }
// };
// TORCH_MODULE(TNet);
//
// // PointNet Classification Model
// struct PointNetImpl : torch::nn::Module {
//     TNet tnet1{nullptr}, tnet2{nullptr};
//     torch::nn::Sequential mlp1{nullptr}, mlp2{nullptr}, classifier{nullptr};
//     int num_classes;
//
//     PointNetImpl(int num_classes_) : num_classes(num_classes_) {
//         tnet1 = register_module("tnet1", TNet(3, 3)); // Input transformation
//         mlp1 = register_module("mlp1", torch::nn::Sequential(
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(3, 64, 1)),
//             torch::nn::BatchNorm1d(64),
//             torch::nn::ReLU(true),
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 64, 1)),
//             torch::nn::BatchNorm1d(64),
//             torch::nn::ReLU(true)
//         ));
//         tnet2 = register_module("tnet2", TNet(64, 64)); // Feature transformation
//         mlp2 = register_module("mlp2", torch::nn::Sequential(
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 64, 1)),
//             torch::nn::BatchNorm1d(64),
//             torch::nn::ReLU(true),
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(64, 128, 1)),
//             torch::nn::BatchNorm1d(128),
//             torch::nn::ReLU(true),
//             torch::nn::Conv1d(torch::nn::Conv1dOptions(128, 1024, 1)),
//             torch::nn::BatchNorm1d(1024),
//             torch::nn::ReLU(true)
//         ));
//         classifier = register_module("classifier", torch::nn::Sequential(
//             torch::nn::Functional([](const torch::Tensor& x) { return torch::max_pool1d(x, x.size(2)).squeeze(2); }),
//             torch::nn::Linear(1024, 512),
//             torch::nn::BatchNorm1d(512),
//             torch::nn::ReLU(true),
//             torch::nn::Dropout(0.3),
//             torch::nn::Linear(512, 256),
//             torch::nn::BatchNorm1d(256),
//             torch::nn::ReLU(true),
//             torch::nn::Dropout(0.3),
//             torch::nn::Linear(256, num_classes_)
//         ));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         // x: [batch, 3, num_points]
//         auto batch_size = x.size(0);
//         auto trans1 = tnet1->forward(x); // [batch, 3, 3]
//         x = torch::bmm(trans1, x); // Apply input transformation
//         x = mlp1->forward(x); // [batch, 64, num_points]
//         auto trans2 = tnet2->forward(x); // [batch, 64, 64]
//         x = torch::bmm(trans2, x); // Apply feature transformation
//         x = mlp2->forward(x); // [batch, 1024, num_points]
//         x = classifier->forward(x); // [batch, num_classes]
//         return {x, trans1};
//     }
//
//     torch::Tensor compute_regularization_loss(const torch::Tensor& trans) {
//         auto batch_size = trans.size(0);
//         auto identity = torch::eye(3, trans.options()).unsqueeze(0).repeat({batch_size, 1, 1});
//         auto diff = torch::bmm(trans, trans.transpose(1, 2)) - identity;
//         return torch::norm(diff, 2).mean();
//     }
// };
// TORCH_MODULE(PointNet);
//
// // Custom Dataset for ModelNet10
// struct ModelNet10Dataset : torch::data::Dataset<ModelNet10Dataset> {
//     std::vector<std::string> pointcloud_paths;
//     std::vector<int64_t> labels;
//     int num_points = 1024;
//
//     ModelNet10Dataset(const std::string& data_dir, bool train) {
//         std::string split = train ? "train" : "test";
//         std::ifstream file(data_dir + "/" + split + "_filelist.txt");
//         std::string line;
//         while (std::getline(file, line)) {
//             std::stringstream ss(line);
//             std::string path;
//             int64_t label;
//             ss >> path >> label;
//             pointcloud_paths.push_back(data_dir + "/" + path);
//             labels.push_back(label);
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         // Load point cloud (text file with x, y, z coordinates)
//         std::ifstream file(pointcloud_paths[index]);
//         std::vector<float> points;
//         std::string line;
//         while (std::getline(file, line) && points.size() / 3 < static_cast<size_t>(num_points)) {
//             std::stringstream ss(line);
//             float x, y, z;
//             ss >> x >> y >> z;
//             points.push_back(x);
//             points.push_back(y);
//             points.push_back(z);
//         }
//         // Subsample or pad to num_points
//         if (points.size() / 3 > static_cast<size_t>(num_points)) {
//             std::random_shuffle-points.begin(), points.end());
//             points.resize(num_points * 3);
//         } else {
//             points.resize(num_points * 3, 0.0f);
//         }
//         auto pointcloud = torch::from_blob(points.data(), {num_points, 3}).t(); // [3, num_points]
//         auto label = torch::tensor(labels[index], torch::kLong);
//         return {pointcloud, label};
//     }
//
//     torch::optional<size_t> size() const override {
//         return pointcloud_paths.size();
//     }
// };
//
// // Main training function
// int main() {
//     // Device configuration
//     torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//     std::cout << "Training on: " << (device.type() == torch::kCUDA ? "GPU" : "CPU") << std::endl;
//
//     // Hyperparameters
//     const int batch_size = 32;
//     const int num_epochs = 50;
//     const float learning_rate = 0.001;
//     const float reg_weight = 0.001; // Regularization weight for T-Net
//     const int num_classes = 10; // ModelNet10
//
//     // Initialize model and optimizer
//     auto model = PointNet(num_classes);
//     model->to(device);
//     auto optimizer = torch::optim::Adam(model->parameters(), learning_rate);
//
//     // Load dataset
//     auto train_dataset = ModelNet10Dataset("data/ModelNet10", true)
//         .map(torch::data::transforms::Stack<>());
//     auto train_loader = torch::data::make_data_loader(
//         train_dataset,
//         torch::data::DataLoaderOptions().batch_size(batch_size).workers(4)
//     );
//
//     // Training loop
//     model->train();
//     for (int epoch = 0; epoch < num_epochs; ++epoch) {
//         float total_loss = 0.0;
//         int correct = 0, total = 0;
//         for (auto& batch : *train_loader) {
//             auto points = batch.data.to(device); // [batch, 3, num_points]
//             auto labels = batch.target.to(device); // [batch]
//
//             optimizer.zero_grad();
//             auto [logits, trans] = model->forward(points);
//             auto cls_loss = torch::nn::functional::cross_entropy(logits, labels);
//             auto reg_loss = model->compute_regularization_loss(trans);
//             auto loss = cls_loss + reg_weight * reg_loss;
//             loss.backward();
//             optimizer.step();
//
//             total_loss += loss.item<float>() * points.size(0);
//             auto pred = logits.argmax(1);
//             correct += pred.eq(labels).sum().item<int64_t>();
//             total += points.size(0);
//         }
//
//         std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
//                   << "], Loss: " << total_loss / total
//                   << ", Accuracy: " << 100.0 * correct / total << "%" << std::endl;
//     }
//
//     // Save model
//     torch::save(model, "pointnet.pt");
//     std::cout << "Model saved to pointnet.pt" << std::endl;
//
//     return 0;
// }




namespace xt::models
{
    PointNet::PointNet(int num_classes, int in_channels)
    {
    }

    PointNet::PointNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PointNet::reset()
    {
    }

    auto PointNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
