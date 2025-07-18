#include <transforms/image/random_crop_near_bbox.h>



// --- Example Main (for testing) ---
// #include "transforms/image/random_crop_near_bbox.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a large dummy image tensor (e.g., 3x500x500)
//     torch::Tensor image = torch::zeros({3, 500, 500});
//
//     // 2. Define a bounding box [xmin, ymin, xmax, ymax]
//     // Let's say it's a 100x80 box somewhere in the middle.
//     torch::Tensor bbox = torch::tensor({200, 220, 300, 300}, torch::kInt32);
//
//     // Let's draw the bbox on the image to visualize it.
//     image.index_put_({torch::indexing::Slice(),
//                      torch::indexing::Slice(bbox[1].item<int>(), bbox[3].item<int>()),
//                      torch::indexing::Slice(bbox[0].item<int>(), bbox[2].item<int>())}, 1.0);
//
//     std::cout << "Original image shape: " << image.sizes() << std::endl;
//     std::cout << "Bounding box: " << bbox << std::endl;
//
//     // --- Example 1: Crop near the bbox ---
//     // Crop a 128x128 region. The crop center will be at most 50 pixels
//     // away from the bbox center.
//     std::cout << "\n--- Cropping 128x128 near bbox (max_dist=50) ---" << std::endl;
//     xt::transforms::image::RandomCropNearBbox cropper({128, 128}, /*max_distance=*/50);
//
//     // Run it a few times to see different crops
//     for (int i = 0; i < 3; ++i) {
//         torch::Tensor cropped_img = std::any_cast<torch::Tensor>(cropper.forward({image, bbox}));
//         std::cout << "Trial " << i << " cropped shape: " << cropped_img.sizes() << std::endl;
//         // In a real scenario, you would save or display the cropped image.
//     }
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomCropNearBbox::RandomCropNearBbox() : RandomCropNearBbox({0, 0}, 0) {}

    RandomCropNearBbox::RandomCropNearBbox(
        std::pair<int, int> crop_size,
        int max_distance)
        : crop_size_(crop_size), max_distance_(max_distance) {

        if (crop_size_.first <= 0 || crop_size_.second <= 0) {
            // This will be caught in forward() if default constructed.
        }
        if (max_distance_ < 0) {
            throw std::invalid_argument("Max distance must be a non-negative integer.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomCropNearBbox::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() < 2) {
            throw std::invalid_argument("RandomCropNearBbox requires an image and a bounding box tensor.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor bbox = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!img.defined() || !bbox.defined()) {
            throw std::invalid_argument("Input image or bbox tensor is not defined.");
        }
        if (crop_size_.first <= 0 || crop_size_.second <= 0) {
            throw std::invalid_argument("RandomCropNearBbox requires a positive crop size.");
        }
        if (bbox.numel() != 4) {
            throw std::invalid_argument("Bounding box must have 4 elements [xmin, ymin, xmax, ymax].");
        }

        // Ensure bbox is on CPU for easy access with .item<T>()
        bbox = bbox.to(torch::kCPU).to(torch::kInt64);

        // --- Calculate Bounding Box and Image Dimensions ---
        auto img_h = img.size(1);
        auto img_w = img.size(2);
        auto crop_h = crop_size_.first;
        auto crop_w = crop_size_.second;

        auto xmin = bbox[0].item<int64_t>();
        auto ymin = bbox[1].item<int64_t>();
        auto xmax = bbox[2].item<int64_t>();
        auto ymax = bbox[3].item<int64_t>();

        // Center of the bounding box
        auto bbox_center_x = (xmin + xmax) / 2;
        auto bbox_center_y = (ymin + ymax) / 2;

        // --- Determine the Valid Sampling Region for the Crop's TOP-LEFT Corner ---
        // Let (crop_x, crop_y) be the top-left corner of the crop.
        // The center of the crop will be at (crop_x + crop_w/2, crop_y + crop_h/2).
        // We require:
        // | (crop_x + crop_w/2) - bbox_center_x | <= max_distance
        // | (crop_y + crop_h/2) - bbox_center_y | <= max_distance

        // This gives us a valid range for crop_x and crop_y:
        int64_t min_crop_x = bbox_center_x - crop_w / 2 - max_distance_;
        int64_t max_crop_x = bbox_center_x - crop_w / 2 + max_distance_;
        int64_t min_crop_y = bbox_center_y - crop_h / 2 - max_distance_;
        int64_t max_crop_y = bbox_center_y - crop_h / 2 + max_distance_;

        // --- Clamp the Sampling Region to be within Image Boundaries ---
        // The crop must be fully contained within the image.
        // The top-left corner (crop_x, crop_y) must satisfy:
        // 0 <= crop_x and crop_x + crop_w <= img_w  => 0 <= crop_x <= img_w - crop_w
        // 0 <= crop_y and crop_y + crop_h <= img_h  => 0 <= crop_y <= img_h - crop_h

        min_crop_x = std::max((int64_t)0, min_crop_x);
        min_crop_y = std::max((int64_t)0, min_crop_y);

        max_crop_x = std::min(img_w - crop_w, max_crop_x);
        max_crop_y = std::min(img_h - crop_h, max_crop_y);

        if (min_crop_x > max_crop_x || min_crop_y > max_crop_y) {
            // This can happen if the image is too small, the bbox is at the edge,
            // and the crop size is large. In this case, we fall back to a standard
            // centered crop on the bbox.
            min_crop_x = bbox_center_x - crop_w / 2;
            min_crop_y = bbox_center_y - crop_h / 2;
            // Clamp this fallback to be safe.
            min_crop_x = std::max((int64_t)0, std::min(min_crop_x, img_w - crop_w));
            min_crop_y = std::max((int64_t)0, std::min(min_crop_y, img_h - crop_h));
            max_crop_x = min_crop_x;
            max_crop_y = min_crop_y;
        }

        // --- Sample a Random Top-Left Corner and Perform Crop ---
        std::uniform_int_distribution<int64_t> x_dist(min_crop_x, max_crop_x);
        std::uniform_int_distribution<int64_t> y_dist(min_crop_y, max_crop_y);

        auto top = y_dist(gen_);
        auto left = x_dist(gen_);

        return img.slice(/*dim=*/1, /*start=*/top, /*end=*/top + crop_h)
                  .slice(/*dim=*/2, /*start=*/left, /*end=*/left + crop_w);
    }

} // namespace xt::transforms::image