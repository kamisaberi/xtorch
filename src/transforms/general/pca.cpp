#include "include/transforms/general/pca.h"
#include <stdexcept>

namespace xt::transforms::general {

    PCA::PCA() = default;

    // The main, recommended constructor
    PCA::PCA(int n_components, std::vector<xt::Module*> pre_transforms)
        : xt::Module(), n_components_(n_components), pre_transforms_(pre_transforms) {
        if (n_components <= 0) {
            throw std::invalid_argument("Number of components (n_components) must be positive.");
        }
    }

    // Constructor to match the user's original request.
    // We'll assume a default or placeholder for n_components.
    PCA::PCA(std::vector<xt::Module*> transforms) : PCA(2, transforms) {
        // Defaults to 2 components if not specified, which is a common choice for visualization.
        // You might want to throw an error here instead to force the user to be explicit.
    }

    // Private helper to run the pre-processing chain
    torch::Tensor PCA::_apply_pre_transforms(torch::Tensor data) {
        torch::Tensor current_data = data;
        for (auto& transform : pre_transforms_) {
            // The output of one transform becomes the input to the next
            current_data = std::any_cast<torch::Tensor>(transform->forward({current_data}));
        }
        return current_data;
    }

    void PCA::fit(torch::Tensor data) {
        // 1. Apply any pre-processing transforms
        data = _apply_pre_transforms(data);

        // 2. Ensure data is a 2D tensor [n_samples, n_features]
        if (data.dim() != 2) {
            throw std::invalid_argument("PCA::fit expects a 2D tensor of shape [n_samples, n_features].");
        }
        int64_t n_features = data.size(1);
        if (n_components_ > n_features) {
            throw std::invalid_argument("n_components cannot be greater than the number of features.");
        }

        // 3. Center the data: subtract the mean from each feature
        mean_ = data.mean(/*dim=*/0);
        data = data - mean_;

        // 4. Compute principal components using SVD.
        // SVD is more numerically stable than computing the covariance matrix.
        // For X = U * S * Vh, the rows of Vh are the principal components.
        auto [U, S, Vh] = xt::linalg::svd(data, /*full_matrices=*/false);

        // 5. Store the top `n_components`
        components_ = Vh.slice(/*dim=*/0, /*start=*/0, /*end=*/n_components_);
        is_fitted_ = true;
    }

    auto PCA::forward(std::initializer_list<std::any> tensors) -> std::any {
        if (!is_fitted_) {
            throw std::runtime_error("PCA transform must be fitted before use. Call the .fit() method first.");
        }

        // Standard input handling
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("PCA::forward received an empty list of tensors.");
        }
        torch::Tensor data = std::any_cast<torch::Tensor>(any_vec[0]);

        // Apply pre-processing transforms
        data = _apply_pre_transforms(data);

        // If data is a batch of images (e.g., [B, C, H, W]), flatten it.
        // If it's a single image, add a batch dim first.
        bool was_3d = false;
        if (data.dim() == 3) {
            was_3d = true;
            data = data.unsqueeze(0);
        }
        if (data.dim() > 2) {
            data = torch::flatten(data, 1);
        }

        // 1. Center the new data using the MEAN FROM THE TRAINING SET
        data = data - mean_;

        // 2. Project the data onto the principal components
        // Result shape: [n_samples, n_components]
        torch::Tensor transformed_data = data.matmul(components_.t());

        // If the original input was a single sample, remove the batch dim
        if (was_3d) {
            transformed_data = transformed_data.squeeze(0);
        }

        return transformed_data;
    }

    bool PCA::is_fitted() const {
        return is_fitted_;
    }

} // namespace xt::transforms::general