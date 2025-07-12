#include "include/transforms/image/random_crop.h"


// --- Example Main (for testing) ---
// #include "transforms/image/random_crop.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a large dummy image tensor (e.g., 3x500x400)
//     torch::Tensor image = torch::randn({3, 500, 400});
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//
//     // --- Example 1: Basic Random Crop ---
//     std::cout << "\n--- Cropping to 224x224 ---" << std::endl;
//     xt::transforms::image::RandomCrop cropper_224({224, 224});
//
//     torch::Tensor cropped_224 = std::any_cast<torch::Tensor>(cropper_224.forward({image}));
//     std::cout << "Cropped shape: " << cropped_224.sizes() << std::endl;
//
//     // --- Example 2: Crop with Padding ---
//     // Padding the original 500x400 image by 50px on each side makes it 600x500.
//     // This increases the chance that the crop includes the original image edges.
//     std::cout << "\n--- Cropping to 256x256 with 50px padding ---" << std::endl;
//     xt::transforms::image::RandomCrop cropper_padded({256, 256}, /*padding=*/50);
//
//     torch::Tensor cropped_padded = std::any_cast<torch::Tensor>(cropper_padded.forward({image}));
//     std::cout << "Cropped shape with padding: " << cropped_padded.sizes() << std::endl;
//
//     // --- Example 3: Cropping a smaller image with pad_if_needed ---
//     torch::Tensor small_image = torch::randn({3, 100, 100});
//     std::cout << "\n--- Cropping small image (100x100) to 128x128 ---" << std::endl;
//     std::cout << "Original small image shape: " << small_image.sizes() << std::endl;
//
//     // This will pad the 100x100 image to 128x128 before "cropping" (returning the whole thing).
//     xt::transforms::image::RandomCrop cropper_pad_needed({128, 128}, /*padding=*/0, /*pad_if_needed=*/true);
//     torch::Tensor cropped_small = std::any_cast<torch::Tensor>(cropper_pad_needed.forward({small_image}));
//     std::cout << "Shape after 'cropping' with pad_if_needed: " << cropped_small.sizes() << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomCrop::RandomCrop() : RandomCrop({0, 0}) {}

    RandomCrop::RandomCrop(
        std::pair<int, int> size,
        int padding,
        bool pad_if_needed,
        double fill)
        : size_(size), padding_(padding), pad_if_needed_(pad_if_needed), fill_(fill) {

        if (size_.first <= 0 || size_.second <= 0) {
            // Note: In the default constructor case, this won't throw, but forward() will.
        }
        if (padding_ < 0) {
            throw std::invalid_argument("Padding must be a non-negative integer.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    std::pair<int, int> RandomCrop::get_crop_params(
        const torch::Tensor& img,
        const std::pair<int, int>& output_size,
        std::mt19937& gen)
    {
        auto img_h = img.size(1);
        auto img_w = img.size(2);
        auto crop_h = output_size.first;
        auto crop_w = output_size.second;

        if (img_w < crop_w || img_h < crop_h) {
            throw std::invalid_argument("Image is smaller than crop size. Consider using 'pad_if_needed'.");
        }

        auto h_range = img_h - crop_h;
        auto w_range = img_w - crop_w;

        std::uniform_int_distribution<int> top_dist(0, h_range);
        std::uniform_int_distribution<int> left_dist(0, w_range);

        return {top_dist(gen), left_dist(gen)};
    }

    auto RandomCrop::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomCrop::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomCrop is not defined.");
        }
        if (size_.first <= 0 || size_.second <= 0) {
            throw std::invalid_argument("RandomCrop requires a positive crop size.");
        }

        // --- Step 1: Optional Padding ---
        if (padding_ > 0) {
            namespace F = torch::nn::functional;
            // Pad is [left, right, top, bottom]
            img = F::pad(img, F::PadFuncOptions({padding_, padding_, padding_, padding_})
                               .mode(torch::kConstant)
                               .value(fill_));
        }

        // --- Step 2: Handle `pad_if_needed` ---
        auto img_h = img.size(1);
        auto img_w = img.size(2);
        auto crop_h = size_.first;
        auto crop_w = size_.second;

        if (pad_if_needed_ && (img_h < crop_h || img_w < crop_w)) {
            auto pad_h = std::max(0L, crop_h - img_h);
            auto pad_w = std::max(0L, crop_w - img_w);
            namespace F = torch::nn::functional;
            // Pad is [left, right, top, bottom]. Here we only pad right and bottom.
            img = F::pad(img, F::PadFuncOptions({0, pad_w, 0, pad_h})
                               .mode(torch::kConstant)
                               .value(fill_));
        }

        // --- Step 3: Get random crop coordinates ---
        auto [top, left] = get_crop_params(img, size_, gen_);
        auto height = size_.first;
        auto width = size_.second;

        // --- Step 4: Perform the crop using tensor slicing ---
        // slice(dim, start, end) where end is exclusive.
        return img.slice(/*dim=*/1, /*start=*/top, /*end=*/top + height)
                  .slice(/*dim=*/2, /*start=*/left, /*end=*/left + width);
    }

} // namespace xt::transforms::image