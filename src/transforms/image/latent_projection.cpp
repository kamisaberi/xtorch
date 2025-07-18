#include <transforms/image/latent_projection.h>


// #include "transforms/general/latent_projection.h"
// #include <iostream>
// #include <memory>
//
// // --- Dummy Encoder for Demonstration ---
// struct EncoderImpl : torch::nn::Module {
//     torch::nn::Sequential layers;
//     EncoderImpl(int latent_dim) {
//         layers = torch::nn::Sequential(
//             torch::nn::Conv2d(3, 16, 3),
//             torch::nn::ReLU(),
//             torch::nn::AdaptiveAvgPool2d(1),
//             torch::nn::Flatten(),
//             torch::nn::Linear(16, latent_dim)
//         );
//         register_module("layers", layers);
//     }
//     torch::Tensor forward(torch::Tensor x) {
//         return layers->forward(x);
//     }
// };
// TORCH_MODULE(Encoder);
//
// int main() {
//     // --- 1. Create and "train" an encoder model ---
//     // In a real scenario, this model would be loaded from a file.
//     int latent_dim = 128;
//     auto encoder = std::make_shared<Encoder>(latent_dim);
//     // Let's assume it's loaded and ready for inference.
//
//     // --- 2. Instantiate the LatentProjection transform, giving it the encoder ---
//     xt::transforms::general::LatentProjection projector(encoder);
//
//     // It's good practice to set the internal model to evaluation mode
//     projector.eval();
//
//     // --- 3. Create a batch of images ---
//     int batch_size = 8;
//     torch::Tensor image_batch = torch::rand({batch_size, 3, 32, 32});
//
//     // --- 4. Apply the transform to project the images into the latent space ---
//     std::any result_any = projector.forward({image_batch});
//     torch::Tensor latent_batch = std::any_cast<torch::Tensor>(result_any);
//
//     // --- 5. Check the output ---
//     std::cout << "Original image batch shape: " << image_batch.sizes() << std::endl;
//     std::cout << "Projected latent batch shape: " << latent_batch.sizes() << std::endl;
//     // Expected output: [8, 128]
//
//     // Now, `latent_batch` could be the input to another model or another transform
//     // like your `LatentInterpolation` transform.
//
//     return 0;
// }

namespace xt::transforms::general
{
    LatentProjection::LatentProjection() = default;

    LatentProjection::LatentProjection(std::shared_ptr<xt::Module> encoder)
        : encoder_(encoder)
    {
        if (!encoder_)
        {
            throw std::invalid_argument("LatentProjection received a null encoder model.");
        }
    }

    auto LatentProjection::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        if (!encoder_)
        {
            throw std::runtime_error("LatentProjection has no encoder model set.");
        }

        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty())
        {
            throw std::invalid_argument("LatentProjection::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined())
        {
            throw std::invalid_argument("Input tensor passed to LatentProjection is not defined.");
        }

        // 2. --- Pass data through the encoder ---
        // We wrap this in a NoGradGuard because feature extraction is typically
        // an inference operation where gradients are not needed. If the user
        // needs to backprop through the encoder, they can manage the gradient
        // context outside of this transform.
        torch::NoGradGuard no_grad;

        torch::Tensor latent_vectors = std::any_cast<torch::Tensor>(encoder_->forward({input_tensor}));

        return latent_vectors;
    }

    void LatentProjection::eval()
    {
        if (encoder_)
        {
            encoder_->eval();
        }
    }

    void LatentProjection::train()
    {
        if (encoder_)
        {
            encoder_->train();
        }
    }
} // namespace xt::transforms::general
