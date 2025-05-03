#include "../../../include/transforms/image/grayscale.h"

namespace xt::transforms::image {

    Grayscale::Grayscale() {
    }

    torch::Tensor Grayscale::operator()(torch::Tensor input) {
        int64_t input_dims = input.dim();
        if (input_dims < 3) {
            throw std::runtime_error("Input tensor must have at least 3 dimensions (e.g., [C, H, W]).");
        }

        // Get channel dimension (assumed as dim 0 or dim 1 for batched)
        int64_t channel_dim = (input_dims == 3) ? 0 : 1;
        int64_t channels = input.size(channel_dim);
        if (channels != 3) {
            throw std::runtime_error("Input tensor must have exactly 3 channels (RGB).");
        }

        // Define grayscale weights (ITU-R 601-2 luma transform)
        auto weights = torch::tensor({0.2989, 0.5870, 0.1140},
                                     torch::TensorOptions().dtype(input.dtype()).device(input.device()));

        // Compute weighted sum across channels
        torch::Tensor gray = (input * weights.view({channels, 1, 1})).sum(channel_dim, true);
        return gray; // Output shape: e.g., [1, H, W] or [N, 1, H, W]
    }





}