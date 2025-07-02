#include <torch/torch.h>
#include <vector>
#include <functional>
#include <iostream>
#include <random>

// ==========================
// ComposeTransform Definition
// ==========================
class ComposeTransform {
public:
    using TransformFunc = std::function<torch::Tensor(torch::Tensor)>;

    // Constructor that accepts an initializer list
    ComposeTransform(std::initializer_list<TransformFunc> init_list)
        : transforms_(init_list) { }

    // Apply all transforms in sequence
    torch::Tensor operator()(torch::Tensor input) const {
        for (const auto& transform : transforms_) {
            input = transform(std::move(input));
        }
        return input;
    }

private:
    std::vector<TransformFunc> transforms_;
};

// ==========================
// Main Program
// ==========================
int main() {
    // Parameters
    int resize_size = 200;
    int crop_size = 100;
    std::vector<double> mean = {0.5, 0.5, 0.5};
    std::vector<double> stddev = {0.5, 0.5, 0.5};

    // Resize function using interpolate
    auto resize_fn = [resize_size](torch::Tensor img) {
        img = img.unsqueeze(0); // Add batch dimension
        img = torch::nn::functional::interpolate(
            img,
            torch::nn::functional::InterpolateFuncOptions()
                .size(std::vector<int64_t>({resize_size, resize_size}))
                .mode(torch::kBilinear)
                .align_corners(false)
        );
        return img.squeeze(0); // Remove batch dimension
    };

    // Random crop function
    auto random_crop_fn = [crop_size](torch::Tensor img) {
        int h = img.size(1);
        int w = img.size(2);
        static std::mt19937 gen{std::random_device{}()};
        std::uniform_int_distribution<int> dist_h(0, h - crop_size);
        std::uniform_int_distribution<int> dist_w(0, w - crop_size);
        int top = dist_h(gen);
        int left = dist_w(gen);
        return img.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(top, top + crop_size),
            torch::indexing::Slice(left, left + crop_size)
        });
    };

    // Convert to float and normalize to [0,1]
    auto to_tensor_fn = [](torch::Tensor img) {
        if (img.dtype() == torch::kUInt8) {
            img = img.to(torch::kFloat32).div(255);
        }
        return img;
    };

    // Normalize using Normalize<> wrapped in a mutable lambda
    auto normalize_fn = torch::data::transforms::Normalize<>(mean, stddev);
    auto norm_fn = [normalize_fn](torch::Tensor img) mutable {
        return normalize_fn(std::move(img));
    };

    // Compose all transforms
    ComposeTransform transform({resize_fn, random_crop_fn, to_tensor_fn, norm_fn});

    // Dummy image tensor (uint8 3x32x32)
    torch::Tensor img = torch::randint(0, 256, {3, 32, 32}, torch::kUInt8);
    std::cout << "Original Image dtype: " << img.dtype() << "\n";

    // Apply transform
    torch::Tensor out = transform(img);
    std::cout << "Transformed Image sizes: " << out.sizes() << ", dtype: " << out.dtype() << "\n";

    return 0;
}
