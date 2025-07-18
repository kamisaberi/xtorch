#include <models/generative_models/gans/star_gan.h>


using namespace std;

//STARGAN GROK


// #include <torch/torch.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include <filesystem>
// #include <iostream>
// #include <vector>
// #include <string>
// #include <random>
//
// // Generator
// struct StarGANGeneratorImpl : torch::nn::Module {
//     StarGANGeneratorImpl(int num_domains) : num_domains_(num_domains) {
//         // Encoder
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1 + num_domains, 64, 4).stride(2).padding(1))); // [batch, 64, 14, 14]
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1))); // [batch, 128, 7, 7]
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1))); // [batch, 256, 3, 3]
//
//         // Decoder
//         deconv3 = register_module("deconv3", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1))); // [batch, 128, 7, 7]
//         deconv2 = register_module("deconv2", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1))); // [batch, 64, 14, 14]
//         deconv1 = register_module("deconv1", torch::nn::ConvTranspose2d(
//             torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1))); // [batch, 1, 28, 28]
//
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
//         bn4 = register_module("bn4", torch::nn::BatchNorm2d(128));
//         bn5 = register_module("bn5", torch::nn::BatchNorm2d(64));
//         relu = register_module("relu", torch::nn::ReLU());
//         tanh = register_module("tanh", torch::nn::Tanh());
//     }
//
//     torch::Tensor forward(torch::Tensor x, torch::Tensor domain_label) {
//         auto batch_size = x.size(0);
//         // Expand domain label to spatial dimensions and concatenate
//         domain_label = domain_label.view({batch_size, num_domains_, 1, 1}).expand({-1, -1, x.size(2), x.size(3)});
//         x = torch::cat({x, domain_label}, 1); // [batch, 1 + num_domains, 28, 28]
//
//         // Encoder
//         auto e1 = relu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//         auto e2 = relu->forward(bn2->forward(conv2->forward(e1))); // [batch, 128, 7, 7]
//         auto e3 = relu->forward(bn3->forward(conv3->forward(e2))); // [batch, 256, 3, 3]
//
//         // Decoder
//         auto d3 = relu->forward(bn4->forward(deconv3->forward(e3))); // [batch, 128, 7, 7]
//         auto d2 = relu->forward(bn5->forward(deconv2->forward(d3))); // [batch, 64, 14, 14]
//         auto d1 = tanh->forward(deconv1->forward(d2)); // [batch, 1, 28, 28]
//
//         return d1;
//     }
//
//     int num_domains_;
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
//     torch::nn::ConvTranspose2d deconv3{nullptr}, deconv2{nullptr}, deconv1{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr}, bn4{nullptr}, bn5{nullptr};
//     torch::nn::ReLU relu{nullptr};
//     torch::nn::Tanh tanh{nullptr};
// };
// TORCH_MODULE(StarGANGenerator);
//
// // Discriminator with Auxiliary Classifier
// struct StarGANDiscriminatorImpl : torch::nn::Module {
//     StarGANDiscriminatorImpl(int num_domains) : num_domains_(num_domains) {
//         conv1 = register_module("conv1", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1))); // [batch, 64, 14, 14]
//         conv2 = register_module("conv2", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1))); // [batch, 128, 7, 7]
//         conv3 = register_module("conv3", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1))); // [batch, 256, 3, 3]
//         conv_src = register_module("conv_src", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(1))); // [batch, 1, 3, 3]
//         conv_cls = register_module("conv_cls", torch::nn::Conv2d(
//             torch::nn::Conv2dOptions(256, num_domains, 3).stride(1).padding(1))); // [batch, num_domains, 3, 3]
//         bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
//         bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
//         bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
//         lrelu = register_module("lrelu", torch::nn::LeakyReLU(
//             torch::nn::LeakyReLUOptions().negative_slope(0.2)));
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x) {
//         x = lrelu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
//         x = lrelu->forward(bn2->forward(conv2->forward(x))); // [batch, 128, 7, 7]
//         x = lrelu->forward(bn3->forward(conv3->forward(x))); // [batch, 256, 3, 3]
//         auto src = conv_src->forward(x); // [batch, 1, 3, 3] (real/fake)
//         auto cls = conv_cls->forward(x); // [batch, num_domains, 3, 3] (domain classification)
//         return {src, cls};
//     }
//
//     int num_domains_;
//     torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr}, conv_src{nullptr}, conv_cls{nullptr};
//     torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
//     torch::nn::LeakyReLU lrelu{nullptr};
// };
// TORCH_MODULE(StarGANDiscriminator);
//
// // Paired Image Dataset with Domain Labels
// struct DomainImageDataset : torch::data::Dataset<DomainImageDataset> {
//     DomainImageDataset(const std::string& img_dir, const std::string& label_file, int num_domains)
//         : num_domains_(num_domains) {
//         // Read label file (format: image_name domain_id)
//         std::ifstream infile(label_file);
//         std::string img_name;
//         int domain_id;
//         while (infile >> img_name >> domain_id) {
//             std::string img_path = img_dir + "/" + img_name;
//             if (std::filesystem::exists(img_path)) {
//                 image_paths_.push_back(img_path);
//                 domain_labels_.push_back(domain_id);
//             }
//         }
//         infile.close();
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
//         // Create one-hot domain label
//         torch::Tensor domain_tensor = torch::zeros({num_domains_}, torch::kFloat32);
//         domain_tensor[domain_labels_[index % image_paths_.size()]] = 1.0;
//
//         return {img_tensor, domain_tensor};
//     }
//
//     torch::optional<size_t> size() const override {
//         return image_paths_.size();
//     }
//
//     std::vector<std::string> image_paths_;
//     std::vector<int> domain_labels_;
//     int num_domains_;
// };
//
// int main() {
//     try {
//         // Set device
//         torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//         std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;
//
//         // Hyperparameters
//         const int num_domains = 3; // e.g., 3 synthetic styles for MNIST
//         const int batch_size = 32;
//         const float lr = 0.0001;
//         const float beta1 = 0.5;
//         const float lambda_cls = 1.0; // Domain classification loss weight
//         const float lambda_rec = 10.0; // Reconstruction loss weight
//         const float lambda_gp = 10.0; // Gradient penalty weight
//
//         // Initialize models
//         StarGANGenerator generator(num_domains);
//         StarGANDiscriminator discriminator(num_domains);
//         generator->to(device);
//         discriminator->to(device);
//
//         // Optimizers
//         torch::optim::Adam g_optimizer(generator->parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));
//         torch::optim::Adam d_optimizer(discriminator->parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));
//
//         // Loss functions
//         auto bce_loss = torch::nn::BCELoss();
//         auto ce_loss = torch::nn::CrossEntropyLoss();
//         auto l1_loss = torch::nn::L1Loss();
//
//         // Gradient penalty (WGAN-GP style)
//         auto compute_gradient_penalty = [&](torch::Tensor real_imgs, torch::Tensor fake_imgs) {
//             auto batch_size = real_imgs.size(0);
//             auto alpha = torch::rand({batch_size, 1, 1, 1}, torch::kFloat32, device);
//             auto interpolates = alpha * real_imgs + (1 - alpha) * fake_imgs;
//             interpolates.set_requires_grad(true);
//             auto [src, _] = discriminator->forward(interpolates);
//             auto gradients = torch::autograd::grad({src}, {interpolates},
//                 torch::TensorOptions().create_vector(true).retain_graph(true))[0];
//             gradients = gradients.view({batch_size, -1});
//             auto gradient_norm = gradients.norm(2, 1);
//             return lambda_gp * ((gradient_norm - 1).pow(2)).mean();
//         };
//
//         // Load dataset
//         auto dataset = DomainImageDataset("./data/images", "./data/labels.txt", num_domains)
//             .map(torch::data::transforms::Stack<>());
//         auto data_loader = torch::data::make_data_loader(
//             dataset, torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
//
//         // Training loop
//         generator->train();
//         discriminator->train();
//         const int num_epochs = 50;
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_int_distribution<> dis(0, num_domains - 1);
//
//         for (int epoch = 0; epoch < num_epochs; ++epoch) {
//             float g_loss_avg = 0.0;
//             float d_loss_avg = 0.0;
//             int batch_count = 0;
//
//             for (auto& batch : *data_loader) {
//                 auto real_imgs = batch.data.to(device);
//                 auto orig_domain = batch.target.to(device);
//                 auto batch_size_current = real_imgs.size(0);
//
//                 // Generate random target domain
//                 torch::Tensor target_domain = torch::zeros({batch_size_current, num_domains}, torch::kFloat32, device);
//                 for (int i = 0; i < batch_size_current; ++i) {
//                     target_domain[i][dis(gen)] = 1.0;
//                 }
//
//                 // Train Discriminator
//                 d_optimizer.zero_grad();
//                 auto [real_src, real_cls] = discriminator->forward(real_imgs);
//                 auto real_label = torch::ones_like(real_src, device);
//                 auto d_loss_real = bce_loss->forward(torch::sigmoid(real_src), real_label);
//
//                 // Domain classification loss for real images
//                 auto cls_loss_real = ce_loss->forward(real_cls.view({-1, num_domains}), orig_domain.argmax(1));
//
//                 // Fake images
//                 auto fake_imgs = generator->forward(real_imgs, target_domain).detach();
//                 auto [fake_src, _] = discriminator->forward(fake_imgs);
//                 auto fake_label = torch::zeros_like(fake_src, device);
//                 auto d_loss_fake = bce_loss->forward(torch::sigmoid(fake_src), fake_label);
//
//                 // Gradient penalty
//                 auto gp = compute_gradient_penalty(real_imgs, fake_imgs);
//
//                 // Total discriminator loss
//                 auto d_loss = d_loss_real + d_loss_fake + cls_loss_real * lambda_cls + gp;
//                 d_loss.backward();
//                 d_optimizer.step();
//
//                 // Train Generator
//                 g_optimizer.zero_grad();
//                 fake_imgs = generator->forward(real_imgs, target_domain);
//                 auto [fake_src_g, fake_cls_g] = discriminator->forward(fake_imgs);
//                 auto g_loss_adv = bce_loss->forward(torch::sigmoid(fake_src_g), real_label);
//
//                 // Domain classification loss for fake images
//                 auto cls_loss_fake = ce_loss->forward(fake_cls_g.view({-1, num_domains}), target_domain.argmax(1));
//
//                 // Reconstruction loss
//                 auto recon_imgs = generator->forward(fake_imgs, orig_domain);
//                 auto rec_loss = l1_loss->forward(recon_imgs, real_imgs);
//
//                 // Total generator loss
//                 auto g_loss = g_loss_adv + lambda_cls * cls_loss_fake + lambda_rec * rec_loss;
//                 g_loss.backward();
//                 g_optimizer.step();
//
//                 d_loss_avg += d_loss.item<float>();
//                 g_loss_avg += g_loss.item<float>();
//                 batch_count++;
//             }
//
//             std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] "
//                       << "Discriminator Loss: " << d_loss_avg / batch_count
//                       << ", Generator Loss: " << g_loss_avg / batch_count << std::endl;
//
//             // Save generated images every 10 epochs
//             if ((epoch + 1) % 10 == 0) {
//                 torch::NoGradGuard no_grad;
//                 auto sample_batch = *data_loader->begin();
//                 auto input_img = sample_batch.data[0].unsqueeze(0).to(device);
//                 for (int d = 0; d < num_domains; ++d) {
//                     torch::Tensor domain = torch::zeros({1, num_domains}, torch::kFloat32, device);
//                     domain[0][d] = 1.0;
//                     auto gen_img = generator->forward(input_img, domain).squeeze().to(torch::kCPU);
//                     cv::Mat img(28, 28, CV_32F, gen_img.data_ptr<float>());
//                     img = (img + 1.0) * 127.5; // Denormalize [-1, 1] to [0, 255]
//                     img.convertTo(img, CV_8U);
//                     cv::imwrite("generated_stargan_epoch_" + std::to_string(epoch + 1) + "_domain_" + std::to_string(d) + ".jpg", img);
//                 }
//             }
//         }
//
//         // Save models
//         torch::save(generator, "stargan_generator.pt");
//         torch::save(discriminator, "stargan_discriminator.pt");
//         std::cout << "Models saved as stargan_generator.pt and stargan_discriminator.pt" << std::endl;
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
    StarGAN::StarGANGeneratorImpl::StarGANGeneratorImpl(int num_domains) : num_domains_(num_domains)
    {
        // Encoder
        conv1 = register_module("conv1", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(1 + num_domains, 64, 4).stride(2).padding(1)));
        // [batch, 64, 14, 14]
        conv2 = register_module("conv2", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
        // [batch, 128, 7, 7]
        conv3 = register_module("conv3", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1)));
        // [batch, 256, 3, 3]

        // Decoder
        deconv3 = register_module("deconv3", torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(256, 128, 4).stride(2).padding(1)));
        // [batch, 128, 7, 7]
        deconv2 = register_module("deconv2", torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(128, 64, 4).stride(2).padding(1)));
        // [batch, 64, 14, 14]
        deconv1 = register_module("deconv1", torch::nn::ConvTranspose2d(
                                      torch::nn::ConvTranspose2dOptions(64, 1, 4).stride(2).padding(1)));
        // [batch, 1, 28, 28]

        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
        bn4 = register_module("bn4", torch::nn::BatchNorm2d(128));
        bn5 = register_module("bn5", torch::nn::BatchNorm2d(64));
        relu = register_module("relu", torch::nn::ReLU());
        tanh = register_module("tanh", torch::nn::Tanh());
    }

    torch::Tensor StarGAN::StarGANGeneratorImpl::forward(torch::Tensor x, torch::Tensor domain_label)
    {
        auto batch_size = x.size(0);
        // Expand domain label to spatial dimensions and concatenate
        domain_label = domain_label.view({batch_size, num_domains_, 1, 1}).expand(
            {-1, -1, x.size(2), x.size(3)});
        x = torch::cat({x, domain_label}, 1); // [batch, 1 + num_domains, 28, 28]

        // Encoder
        auto e1 = relu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
        auto e2 = relu->forward(bn2->forward(conv2->forward(e1))); // [batch, 128, 7, 7]
        auto e3 = relu->forward(bn3->forward(conv3->forward(e2))); // [batch, 256, 3, 3]

        // Decoder
        auto d3 = relu->forward(bn4->forward(deconv3->forward(e3))); // [batch, 128, 7, 7]
        auto d2 = relu->forward(bn5->forward(deconv2->forward(d3))); // [batch, 64, 14, 14]
        auto d1 = tanh->forward(deconv1->forward(d2)); // [batch, 1, 28, 28]

        return d1;
    }


    StarGAN::StarGANDiscriminatorImpl::StarGANDiscriminatorImpl(int num_domains) : num_domains_(num_domains)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(1, 64, 4).stride(2).padding(1)));
        // [batch, 64, 14, 14]
        conv2 = register_module("conv2", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(64, 128, 4).stride(2).padding(1)));
        // [batch, 128, 7, 7]
        conv3 = register_module("conv3", torch::nn::Conv2d(
                                    torch::nn::Conv2dOptions(128, 256, 4).stride(2).padding(1)));
        // [batch, 256, 3, 3]
        conv_src = register_module("conv_src", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(256, 1, 3).stride(1).padding(1)));
        // [batch, 1, 3, 3]
        conv_cls = register_module("conv_cls", torch::nn::Conv2d(
                                       torch::nn::Conv2dOptions(256, num_domains, 3).stride(1).padding(1)));
        // [batch, num_domains, 3, 3]
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(128));
        bn3 = register_module("bn3", torch::nn::BatchNorm2d(256));
        lrelu = register_module("lrelu", torch::nn::LeakyReLU(
                                    torch::nn::LeakyReLUOptions().negative_slope(0.2)));
    }

    std::tuple<torch::Tensor, torch::Tensor> StarGAN::StarGANDiscriminatorImpl::forward(torch::Tensor x)
    {
        x = lrelu->forward(bn1->forward(conv1->forward(x))); // [batch, 64, 14, 14]
        x = lrelu->forward(bn2->forward(conv2->forward(x))); // [batch, 128, 7, 7]
        x = lrelu->forward(bn3->forward(conv3->forward(x))); // [batch, 256, 3, 3]
        auto src = conv_src->forward(x); // [batch, 1, 3, 3] (real/fake)
        auto cls = conv_cls->forward(x); // [batch, num_domains, 3, 3] (domain classification)
        return {src, cls};
    }


    StarGAN::StarGAN(int num_classes, int in_channels)
    {
    }

    StarGAN::StarGAN(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void StarGAN::reset()
    {
    }

    auto StarGAN::forward(std::initializer_list<std::any> tensors) -> std::any
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
