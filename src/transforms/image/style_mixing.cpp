#include <transforms/image/style_mixing.h>

// #include "transforms/general/style_mixing.h"
// #include <iostream>
//
// // --- Dummy Mapping Network for Demonstration ---
// struct MappingNetwork : torch::nn::Module {
//     torch::nn::Linear layer;
//     MappingNetwork(int z_dim, int w_dim) : layer(z_dim, w_dim) {
//         register_module("layer", layer);
//     }
//     torch::Tensor forward(torch::Tensor z) {
//         return layer->forward(z);
//     }
// };
//
// int main() {
//     // --- 1. Setup ---
//     int batch_size = 4;
//     int z_dim = 512;
//     int w_dim = 512;
//     int n_layers = 14; // For a 256x256 StyleGAN generator
//
//     // --- 2. Create the transform ---
//     xt::transforms::general::StyleMixing mixer(0.9f, n_layers);
//
//     // --- 3. Simulate a training step ---
//     // Create a dummy mapping network
//     MappingNetwork mapping_network(z_dim, w_dim);
//
//     // Generate two batches of random latent codes (z-space)
//     torch::Tensor z1 = torch::randn({batch_size, z_dim});
//     torch::Tensor z2 = torch::randn({batch_size, z_dim});
//
//     // Map them to the style space (w-space)
//     torch::Tensor w1 = mapping_network.forward(z1);
//     torch::Tensor w2 = mapping_network.forward(z2);
//
//     std::cout << "Shape of input style vectors (w1, w2): " << w1.sizes() << std::endl;
//
//     // --- 4. Apply the StyleMixing transform ---
//     // This takes the two style vectors and creates the mixed input for the generator.
//     std::any result_any = mixer.forward({w1, w2});
//     torch::Tensor mixed_styles = std::any_cast<torch::Tensor>(result_any);
//
//     // --- 5. Check the output ---
//     std::cout << "Shape of mixed style tensor for generator: " << mixed_styles.sizes() << std::endl;
//     // Expected output: [4, 14, 512]
//
//     // The `mixed_styles` tensor is now ready to be fed into the synthesis network of StyleGAN.
//     // Each of the 4 samples in the batch will have a different random crossover point,
//     // mixing the styles from `w1` and `w2` at different layers.
//
//     return 0;
// }

namespace xt::transforms::general {

    // StyleGAN2 for 256x256 has 14 style inputs (2 per resolution from 4x4 to 256x256)
    StyleMixing::StyleMixing() : p_(0.9f), n_layers_(14) {}

    StyleMixing::StyleMixing(float p, int n_layers) : p_(p), n_layers_(n_layers) {
        if (p_ < 0.0f || p_ > 1.0f) {
            throw std::invalid_argument("StyleMixing probability must be between 0.0 and 1.0.");
        }
        if (n_layers_ <= 0) {
            throw std::invalid_argument("Number of layers must be positive.");
        }
    }

    auto StyleMixing::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- 1. Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 2) {
            throw std::invalid_argument("StyleMixing::forward expects two tensors: primary and secondary style vectors.");
        }

        torch::Tensor w1 = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor w2 = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!w1.defined() || !w2.defined()) {
            throw std::invalid_argument("Input style vectors are not defined.");
        }

        // Ensure inputs are broadcastable to [B, n_layers, D]
        if (w1.dim() == 2) { // Input is [B, D], needs to be broadcast to all layers
            w1 = w1.unsqueeze(1).repeat({1, n_layers_, 1});
        }
        if (w2.dim() == 2) {
            w2 = w2.unsqueeze(1).repeat({1, n_layers_, 1});
        }

        if (w1.dim() != 3 || w1.size(1) != n_layers_ || w2.dim() != 3 || w2.size(1) != n_layers_) {
            throw std::invalid_argument("Style vectors must have shape [B, D] or [B, n_layers, D].");
        }

        // --- 2. Decide whether to apply mixing ---
        if (torch::rand({1}).item<float>() > p_) {
            // If not mixing, just return the primary style vector.
            return w1;
        }

        // --- 3. Generate Crossover Point and Mix Styles ---
        // Choose a random layer index to be the crossover point.
        int64_t crossover_point = torch::randint(1, n_layers_, {1}).item<int64_t>();

        // Create a mask for selecting which styles to use.
        // The mask will be [1, n_layers, 1] for broadcasting.
        auto layer_indices = torch::arange(0, n_layers_, w1.options().dtype(torch::kLong)).view({1, -1, 1});

        // The mask is `true` for `w1` and `false` for `w2`.
        torch::Tensor mask = (layer_indices < crossover_point);

        // `torch::where` selects elements from `w1` where the mask is true,
        // and from `w2` where it is false. This is a highly efficient way to do the mix.
        torch::Tensor mixed_w = torch::where(mask, w1, w2);

        return mixed_w;
    }

} // namespace xt::transforms::general