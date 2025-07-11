#include "include/transforms/image/fancy_pca.h"




// #include "transforms/image/fancy_pca.h"
// #include <iostream>
//
// int main() {
//     // --- 1. Define the pre-computed PCA statistics for ImageNet ---
//     // These are standard values used in many implementations.
//     // Eigenvectors (principal components of color)
//     torch::Tensor eigenvectors = torch::tensor({
//         {-0.5675,  0.7192,  0.4009},
//         {-0.5808, -0.0045, -0.8140},
//         {-0.5836, -0.6948,  0.4203}
//     }, torch::kFloat32);
//
//     // Eigenvalues (standard deviation of color along each component)
//     torch::Tensor eigenvalues = torch::tensor({0.2175, 0.0188, 0.0045}, torch::kFloat32);
//
//     // --- 2. Create a dummy color image ---
//     torch::Tensor image = torch::rand({3, 224, 224});
//
//     std::cout << "Original image mean per channel: " << image.mean({1, 2}) << std::endl;
//
//     // --- 3. Instantiate the FancyPCA transform ---
//     xt::transforms::image::FancyPCA color_jitter(eigenvectors, eigenvalues);
//
//     // --- 4. Apply the transform ---
//     std::any result_any = color_jitter.forward({image});
//     torch::Tensor jittered_image = std::any_cast<torch::Tensor>(result_any);
//
//     // --- 5. Check the output ---
//     std::cout << "Jittered image mean per channel: " << jittered_image.mean({1, 2}) << std::endl;
//     std::cout << "Image shape remains: " << jittered_image.sizes() << std::endl;
//
//     // The mean values of the channels will have shifted slightly and randomly,
//     // simulating a natural lighting change.
//
//     return 0;
// }

namespace xt::transforms::image {

    FancyPCA::FancyPCA() = default;

    FancyPCA::FancyPCA(torch::Tensor eigenvectors, torch::Tensor eigenvalues, double alpha_std)
        : eigenvectors_(eigenvectors), eigenvalues_(eigenvalues), alpha_std_(alpha_std) {

        if (!eigenvectors_.defined() || !eigenvalues_.defined()) {
            throw std::invalid_argument("Eigenvectors and eigenvalues must be defined tensors.");
        }
        if (eigenvectors_.sizes() != torch::IntArrayRef({3, 3})) {
            throw std::invalid_argument("Eigenvectors must be a 3x3 tensor.");
        }
        if (eigenvalues_.sizes() != torch::IntArrayRef({3})) {
            throw std::invalid_argument("Eigenvalues must be a 1D tensor of size 3.");
        }
    }

    auto FancyPCA::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("FancyPCA::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) {
            throw std::invalid_argument("Input tensor passed to FancyPCA is not defined.");
        }
        if (image.dim() != 3 || image.size(0) != 3) {
            throw std::invalid_argument("FancyPCA expects a 3-channel image tensor (C, H, W).");
        }

        // 2. --- Generate Random Strengths (Alphas) ---
        // Sample a random strength for each principal component from a normal distribution.
        // The standard deviation of this distribution is given by the eigenvalues.
        torch::Tensor alphas = torch::randn({3}, image.options()) * eigenvalues_;

        // Optionally, scale the overall strength of the effect randomly.
        if (alpha_std_ > 0.0) {
            alphas *= torch::rand({1}, image.options()).item<float>() * alpha_std_;
        }

        // 3. --- Calculate the Color Offset ---
        // The final color offset to add to the image is a linear combination of the
        // eigenvectors, weighted by the random alphas.
        // (3x3).T() * (3) -> (3x3) * (3x1) -> (3x1)
        torch::Tensor offset = eigenvectors_.t().matmul(alphas.unsqueeze(1)).squeeze(1);

        // 4. --- Add Offset to the Image ---
        // We need to reshape the offset from [3] to [3, 1, 1] so that it can be
        // broadcast correctly across the spatial dimensions (H, W) of the image.
        torch::Tensor offset_reshaped = offset.view({3, 1, 1});

        torch::Tensor jittered_image = image + offset_reshaped;

        // 5. --- Clamp to Valid Range ---
        // Ensure pixel values stay within the valid [0, 1] range.
        jittered_image = torch::clamp(jittered_image, 0.0, 1.0);

        return jittered_image;
    }

} // namespace xt::transforms::image