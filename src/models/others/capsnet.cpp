#include "include/models/others/capsnet.h"


using namespace std;


// #include <torch/torch.h>
// #include <iostream>
// #include <vector>
// #include <memory>
//
// using namespace torch;
//
// // Squash activation function for capsules
// torch::Tensor squash(const torch::Tensor& input) {
//     auto squared_norm = (input * input).sum(-1, true);
//     auto scale = squared_norm / (1 + squared_norm);
//     return scale * input / torch::sqrt(squared_norm + 1e-8);
// }
//
// // Capsule Layer
// struct CapsuleLayerImpl : nn::Module {
//     int in_capsules, out_capsules, in_dim, out_dim;
//     torch::Tensor W; // Weight matrix for transformations
//
//     CapsuleLayerImpl(int in_capsules_, int out_capsules_, int in_dim_, int out_dim_)
//         : in_capsules(in_capsules_),
//           out_capsules(out_capsules_),
//           in_dim(in_dim_),
//           out_dim(out_dim_) {
//         W = register_parameter(
//             "W",
//             torch::randn({out_capsules, in_capsules, out_dim, in_dim})
//             * 0.01
//         );
//     }
//
//     torch::Tensor forward(const torch::Tensor& u, int num_iterations = 3) {
//         // u: [batch, in_capsules, in_dim]
//         auto batch_size = u.size(0);
//         // Transform inputs: u_hat = W * u
//         auto u_reshaped = u.unsqueeze(0).unsqueeze(3); // [1, batch, in_caps, in_dim, 1]
//         auto W_expanded = W.unsqueeze(1); // [out_caps, 1, in_caps, out_dim, in_dim]
//         auto u_hat = torch::matmul(W_expanded, u_reshaped); // [out_caps, batch, in_caps, out_dim, 1]
//         u_hat = u_hat.squeeze(-1).permute({1, 2, 0, 3}); // [batch, in_caps, out_caps, out_dim]
//
//         // Initialize coupling coefficients
//         auto b = torch::zeros({batch_size, in_capsules, out_capsules}, u.options());
//
//         // Dynamic routing
//         torch::Tensor v;
//         for (int i = 0; i < num_iterations; ++i) {
//             auto c = torch::softmax(b, 2); // [batch, in_caps, out_caps]
//             auto s = torch::einsum("bic,bico->bco", {c, u_hat}); // [batch, out_caps, out_dim]
//             v = squash(s); // [batch, out_caps, out_dim]
//             if (i < num_iterations - 1) {
//                 auto delta_b = torch::einsum("bco,bico->bic", {v, u_hat});
//                 b = b + delta_b;
//             }
//         }
//         return v; // [batch, out_caps, out_dim]
//     }
// };
// TORCH_MODULE(CapsuleLayer);
//
// // CapsNet Architecture
// struct CapsNetImpl : nn::Module {
//     nn::Conv2d conv1{nullptr};
//     nn::Conv2d primary_caps_conv{nullptr};
//     CapsuleLayer digit_caps{nullptr};
//     float m_plus = 0.9, m_minus = 0.1, lambda_ = 0.5;
//
//     CapsNetImpl() {
//         conv1 = register_module(
//             "conv1",
//             nn::Conv2d(nn::Conv2dOptions(1, 256, 9).stride(1))
//         );
//         primary_caps_conv = register_module(
//             "primary_caps_conv",
//             nn::Conv2d(nn::Conv2dOptions(256, 32 * 8, 9).stride(2))
//         );
//         digit_caps = register_module(
//             "digit_caps",
//             CapsuleLayer(32 * 6 * 6, 10, 8, 16)
//         );
//     }
//
//     torch::Tensor forward(const torch::Tensor& x) {
//         // x: [batch, 1, 28, 28]
//         auto out = torch::relu(conv1->forward(x)); // [batch, 256, 20, 20]
//         out = primary_caps_conv->forward(out); // [batch, 32*8, 6, 6]
//         out = out.view({out.size(0), 32 * 6 * 6, 8}); // [batch, 32*6*6, 8]
//         out = squash(out); // Primary capsules
//         out = digit_caps->forward(out); // [batch, 10, 16]
//         return out;
//     }
//
//     torch::Tensor margin_loss(const torch::Tensor& v, const torch::Tensor& labels) {
//         auto batch_size = v.size(0);
//         auto v_norm = torch::sqrt((v * v).sum(-1) + 1e-8); // [batch, 10]
//         auto left = torch::relu(m_plus - v_norm).pow(2);
//         auto right = torch::relu(v_norm - m_minus).pow(2);
//         auto loss = labels * left + lambda_ * (1 - labels) * right;
//         return loss.sum(1).mean();
//     }
// };
// TORCH_MODULE(CapsNet);
//
// // Main training function
// int main() {
//     // Device configuration
//     torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//     std::cout << "Training on: " << (device.type() == torch::kCUDA ? "GPU" : "CPU") << std::endl;
//
//     // Hyperparameters
//     const int batch_size = 128;
//     const int num_epochs = 10;
//     const float learning_rate = 0.001;
//
//     // Initialize model and optimizer
//     auto model = CapsNet();
//     model->to(device);
//     auto optimizer = torch::optim::Adam(model->parameters(), learning_rate);
//
//     // Load MNIST dataset
//     auto train_dataset = torch::data::datasets::MNIST("./data")
//         .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
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
//
//         for (auto& batch : *train_loader) {
//             auto data = batch.data.to(device);
//             auto target = batch.target.to(device);
//
//             // One-hot encode labels
//             auto labels = torch::zeros({target.size(0), 10}, data.options()).scatter_(1, target.view({-1, 1}), 1);
//
//             optimizer.zero_grad();
//             auto output = model->forward(data);
//             auto loss = model->margin_loss(output, labels);
//             loss.backward();
//             optimizer.step();
//
//             total_loss += loss.item<float>() * data.size(0);
//
//             // Compute accuracy
//             auto v_norm = torch::sqrt((output * output).sum(-1) + 1e-8);
//             auto pred = v_norm.argmax(1);
//             correct += pred.eq(target).sum().item<int64_t>();
//             total += data.size(0);
//         }
//
//         std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs
//                   << "], Loss: " << total_loss / total
//                   << ", Accuracy: " << 100.0 * correct / total << "%" << std::endl;
//     }
//
//     // Save model
//     torch::save(model, "capsnet_mnist.pt");
//     std::cout << "Model saved to capsnet_mnist.pt" << std::endl;
//
//     return 0;
// }

namespace xt::models
{
    CapsNet::CapsNet(int num_classes, int in_channels)
    {
    }

    CapsNet::CapsNet(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void CapsNet::reset()
    {
    }

    auto CapsNet::forward(std::initializer_list<std::any> tensors) -> std::any
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
