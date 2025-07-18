#include <models/generative_models/others/pixel_rnn.h>


using namespace std;

//PIXELRNN GROK

// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
//
// // RowLSTM Layer (simplified for grayscale)
// struct RowLSTMImpl : torch::nn::Module {
//     RowLSTMImpl(int input_size, int hidden_size) : hidden_size_(hidden_size) {
//         // Input gate
//         Wi = register_parameter("Wi", torch::randn({input_size, hidden_size}) * 0.01);
//         Ui = register_parameter("Ui", torch::randn({hidden_size, hidden_size}) * 0.01);
//         bi = register_parameter("bi", torch::zeros({hidden_size}));
//
//         // Forget gate
//         Wf = register_parameter("Wf", torch::randn({input_size, hidden_size}) * 0.01);
//         Uf = register_parameter("Uf", torch::randn({hidden_size, hidden_size}) * 0.01);
//         bf = register_parameter("bf", torch::zeros({hidden_size}));
//
//         // Cell gate
//         Wc = register_parameter("Wc", torch::randn({input_size, hidden_size}) * 0.01);
//         Uc = register_parameter("Uc", torch::randn({hidden_size, hidden_size}) * 0.01);
//         bc = register_parameter("bc", torch::zeros({hidden_size}));
//
//         // Output gate
//         Wo = register_parameter("Wo", torch::randn({input_size, hidden_size}) * 0.01);
//         Uo = register_parameter("Uo", torch::randn({hidden_size, hidden_size}) * 0.01);
//         bo = register_parameter("bo", torch::zeros({hidden_size}));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor h_prev, torch::Tensor c_prev) {
//         // x: [batch, input_size], h_prev: [batch, hidden_size], c_prev: [batch, hidden_size]
//         auto i = torch::sigmoid(torch::matmul(x, Wi) + torch::matmul(h_prev, Ui) + bi); // Input gate
//         auto f = torch::sigmoid(torch::matmul(x, Wf) + torch::matmul(h_prev, Uf) + bf); // Forget gate
//         auto g = torch::tanh(torch::matmul(x, Wc) + torch::matmul(h_prev, Uc) + bc); // Cell gate
//         auto o = torch::sigmoid(torch::matmul(x, Wo) + torch::matmul(h_prev, Uo) + bo); // Output gate
//
//         auto c = f * c_prev + i * g; // Cell state
//         auto h = o * torch::tanh(c); // Hidden state
//
//         return {h, c};
//     }
//
//     int hidden_size_;
//     torch::Tensor Wi, Ui, bi, Wf, Uf, bf, Wc, Uc, bc, Wo, Uo, bo;
// };
// TORCH_MODULE(RowLSTM);
//
// // PixelRNN Model
// struct PixelRNNImpl : torch::nn::Module {
//     PixelRNNImpl(int num_levels, int hidden_size = 128, int num_layers = 2)
//         : num_levels_(num_levels), hidden_size_(hidden_size), num_layers_(num_layers) {
//         // Initial conv to embed input pixels
//         input_conv = register_module("input_conv", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, hidden_size, 1).stride(1)));
//
//         // RowLSTM layers
//         for (int i = 0; i < num_layers; ++i) {
//             lstms->push_back("lstm_" + std::to_string(i),
//                              RowLSTM(hidden_size, hidden_size));
//         }
//         lstms = register_module("lstms", lstms);
//
//         // Output layer
//         output_fc = register_module("output_fc", torch::nn::Linear(hidden_size, num_levels));
//
//         relu = register_module("relu", torch::nn::ReLU());
//     }
//
//     torch::Tensor forward(torch::Tensor x) {
//         // x: [batch, 1, height, width]
//         auto batch_size = x.size(0);
//         auto height = x.size(2);
//         auto width = x.size(3);
//
//         // Initialize hidden and cell states
//         std::vector<std::vector<torch::Tensor>> h(num_layers_,
//             std::vector<torch::Tensor>(width, torch::zeros({batch_size, hidden_size_}, x.device())));
//         std::vector<std::vector<torch::Tensor>> c(num_layers_,
//             std::vector<torch::Tensor>(width, torch::zeros({batch_size, hidden_size_}, x.device())));
//
//         // Embed input
//         x = relu->forward(input_conv->forward(x)); // [batch, hidden_size, height, width]
//
//         // Output logits
//         auto logits = torch::zeros({batch_size, num_levels_, height, width}, x.device());
//
//         // Process each pixel
//         for (int i = 0; i < height; ++i) {
//             for (int j = 0; j < width; ++j) {
//                 auto pixel_embed = x.index({torch::indexing::Slice(), torch::indexing::Slice(), i, j}); // [batch, hidden_size]
//
//                 // Run through LSTM layers
//                 auto h_curr = pixel_embed;
//                 for (int l = 0; l < num_layers_; ++l) {
//                     auto [h_new, c_new] = lstms[l]->as<RowLSTM>()->forward(
//                         h_curr,
//                         j > 0 ? h[l][j - 1] : torch::zeros_like(h_curr),
//                         j > 0 ? c[l][j - 1] : torch::zeros_like(h_curr)
//                     );
//                     h[l][j] = h_new;
//                     c[l][j] = c_new;
//                     h_curr = h_new;
//                 }
//
//                 // Predict logits for current pixel
//                 logits.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), i, j},
//                                  output_fc->forward(h_curr));
//             }
//         }
//
//         return logits; // [batch, num_levels, height, width]
//     }
//
//     int num_levels_, hidden_size_, num_layers_;
//     torch::nn::Conv2d input_conv{nullptr};
//     torch::nn::Sequential lstms{nullptr};
//     torch::nn::Linear output_fc{nullptr};
//     torch::nn::ReLU relu{nullptr};
// };
// TORCH_MODULE(PixelRNN);
//
// // Custom Dataset for Quantized Grayscale Images
// struct QuantizedImageDataset : torch::data::Dataset<QuantizedImageDataset> {
//     QuantizedImageDataset(const std::string& img_dir, int num_levels) : num_levels_(num_levels) {
//         for (const auto& entry : std::filesystem::directory_iterator(img_dir)) {
//             if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
//                 image_paths_.push_back(entry.path().string());
//             }
//         }
//     }
//
//     torch::data::Example<> get(size_t index) override {
//         cv::Mat image = cv::imread(image_paths_[index % image_paths_.size()], cv::IMREAD_GRAYSCALE);
//         if (image.empty()) {
//             throw std::runtime_error("Failed to load image: " + image_paths_[index % image_paths_.size()]);
//         }
//         image.convertTo(image, CV_32F, 1.0 / 255.0); // Normalize to [0, 1]
//
//         // Quantize to num_levels
//         cv::Mat quantized;
//         image.convertTo(quantized, CV_32F, num_levels_ - 1);
//         quantized = quantized.round();
//         torch::Tensor img_tensor = torch::from_blob(quantized.data, {1, image.rows, image.cols}, torch::kFloat32);
//         torch::Tensor label_tensor = torch::from_blob(quantized.data, {image.rows, image.cols}, torch::kInt64);
//
//         return {img_tensor, label_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_;
//     int num_levels_;
// };
//
// // Generate samples from PixelRNN
// torch::Tensor generate_samples(PixelRNN& model, int num_samples, int height, int width, int num_levels, torch::Device device) {
//     torch::NoGradGuard no_grad;
//     auto samples = torch::zeros({num_samples, 1, height, width}, torch::kFloat32, device);
//
//     for (int i = 0; i < height; ++i) {
//         for (int j = 0; j < width; ++j) {
//             auto logits = model->forward(samples); // [num_samples, num_levels, height, width]
//             auto probs = torch::softmax(logits.slice(2, i, i + 1).slice(3, j, j + 1), 1);
//             probs = probs.squeeze(3).squeeze(2); // [num_samples, num_levels]
//             auto pixel_values = torch::multinomial(probs, 1).to(torch::kFloat32); // [num_samples, 1]
//             samples.index_put_({torch::indexing::Slice(), 0, i, j}, pixel_values / (num_levels - 1));
//         }
//     }
//
//     return samples;
// }
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int num_levels = 4; // Quantized pixel levels (e.g., 0, 1/3, 2/3, 1)
//         const int batch_size = 16; // Smaller due to RNN memory usage
//         const int hidden_size = 128;
//         const int num_layers = 2;
//         const float lr = 0.001;
//         const int num_epochs = 20;
//
//         // Initialize model
//         PixelRNN model(num_levels, hidden_size, num_layers);
//         model->to(device);
//
//         // Optimizer
//         torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(lr));
//
//         // Loss function
//         auto ce_loss = torch::nn::CrossEntropyLoss();
//
//         // Load dataset
//         auto dataset = QuantizedImageDataset("./data/images", num_levels)
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
//                 auto labels = batch.target.to(device); // [batch, 28, 28]
//
//                 optimizer.zero_grad();
//                 auto logits = model->forward(images); // [batch, num_levels, 28, 28]
//                 // Reshape for cross-entropy: [batch * 28 * 28, num_levels]
//                 auto loss = ce_loss->forward(logits.permute({0, 2, 3, 1}).reshape({-1, num_levels}), labels.reshape(-1));
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
//             // Generate samples every 5 epochs
//             if ((epoch + 1) % 5 == 0) {
//                 auto samples = generate_samples(model, 1, 28, 28, num_levels, device).squeeze().to(torch::kCPU);
//                 cv::Mat img(28, 28, CV_32F, samples.data_ptr<float>());
//                 img.convertTo(img, CV_8U, 255.0);
//                 cv::imwrite("generated_pixelrnn_epoch_" + std::to_string(epoch + 1) + ".jpg", img);
//             }
//         }
//
//         // Save model
//         torch::save(model, "pixelrnn.pt");
//         std::cout << "Model saved as pixelrnn.pt" << std::endl;
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
    PixelRNN::PixelRNN(int num_classes, int in_channels)
    {
    }

    PixelRNN::PixelRNN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void PixelRNN::reset()
    {
    }

    auto PixelRNN::forward(std::initializer_list<std::any> tensors) -> std::any
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
