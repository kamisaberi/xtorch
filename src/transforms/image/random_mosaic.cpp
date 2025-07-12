#include "include/transforms/image/random_mosaic.h"


// --- Example Main (for testing) ---
// #include "transforms/image/random_mosaic.h"
// #include "utils/image_conversion.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
//
// // Helper to draw bboxes for visualization
// void draw_bboxes(cv::Mat& img, const torch::Tensor& bboxes) {
//     for (int i = 0; i < bboxes.size(0); ++i) {
//         auto box = bboxes[i];
//         cv::Point p1(box[0].item<int>(), box[1].item<int>());
//         cv::Point p2(box[2].item<int>(), box[3].item<int>());
//         cv::rectangle(img, p1, p2, {0, 255, 0}, 2); // Green boxes
//     }
// }
//
// int main() {
//     // 1. Create 4 dummy images of different sizes and colors
//     torch::Tensor img1 = torch::full({3, 300, 400}, 0.2); // Dark gray
//     torch::Tensor img2 = torch::full({3, 500, 250}, 0.4); // Medium gray
//     torch::Tensor img3 = torch::full({3, 480, 640}, 0.6); // Light gray
//     torch::Tensor img4 = torch::full({3, 350, 350}, 0.8); // Very light gray
//
//     // 2. Create dummy bounding boxes for each image
//     // Format: [xmin, ymin, xmax, ymax, class_id]
//     torch::Tensor bboxes1 = torch::tensor({{10, 20, 80, 100, 0}}, torch::kFloat32);
//     torch::Tensor bboxes2 = torch::tensor({{50, 50, 150, 150, 1}}, torch::kFloat32);
//     torch::Tensor bboxes3 = torch::tensor({{200, 100, 400, 300, 2}}, torch::kFloat32);
//     torch::Tensor bboxes4 = torch::tensor({{100, 100, 200, 200, 3}}, torch::kFloat32);
//
//     std::cout << "--- Applying Mosaic ---" << std::endl;
//
//     // 3. Define the transform for a 640x640 output
//     xt::transforms::image::RandomMosaic mosaic_transform({640, 640});
//
//     // 4. Apply the transform
//     std::any result = mosaic_transform.forward({img1, bboxes1, img2, bboxes2, img3, bboxes3, img4, bboxes4});
//     auto result_pair = std::any_cast<std::pair<torch::Tensor, torch::Tensor>>(result);
//
//     torch::Tensor mosaic_img = result_pair.first;
//     torch::Tensor combined_bboxes = result_pair.second;
//
//     std::cout << "Mosaic image shape: " << mosaic_img.sizes() << std::endl;
//     std::cout << "Combined bboxes shape: " << combined_bboxes.sizes() << std::endl;
//     std::cout << "Combined bboxes:\n" << combined_bboxes << std::endl;
//
//     // 5. Save the result for visualization
//     cv::Mat mosaic_mat = xt::utils::image::tensor_to_mat_8u(mosaic_img);
//     draw_bboxes(mosaic_mat, combined_bboxes);
//     cv::imwrite("mosaic_image.png", mosaic_mat);
//     std::cout << "Saved mosaic_image.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomMosaic::RandomMosaic() : RandomMosaic({640, 640}, 0.5) {}

    RandomMosaic::RandomMosaic(std::pair<int, int> output_size, double fill)
        : output_size_(output_size), fill_(fill) {

        if (output_size_.first <= 0 || output_size_.second <= 0) {
            throw std::invalid_argument("Output size must be positive.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomMosaic::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Parsing and Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() != 8) {
            throw std::invalid_argument("Mosaic::forward expects 8 inputs: {img1, bbox1, ... img4, bbox4}.");
        }
        std::vector<torch::Tensor> images;
        std::vector<torch::Tensor> bboxes;
        for (size_t i = 0; i < any_vec.size(); i += 2) {
            images.push_back(std::any_cast<torch::Tensor>(any_vec[i]));
            bboxes.push_back(std::any_cast<torch::Tensor>(any_vec[i + 1]));
        }

        auto output_h = output_size_.first;
        auto output_w = output_size_.second;

        // --- Determine Random Center Point ---
        std::uniform_int_distribution<> x_dist(output_w / 4, output_w * 3 / 4);
        std::uniform_int_distribution<> y_dist(output_h / 4, output_h * 3 / 4);
        auto center_x = x_dist(gen_);
        auto center_y = y_dist(gen_);

        // --- Create Mosaic Canvas ---
        torch::Tensor mosaic_image = torch::full({images[0].size(0), output_h, output_w}, fill_, images[0].options());

        // Store the top-left corner coordinates for each image placed on the mosaic
        std::vector<std::pair<int, int>> paste_coords(4);

        // --- Place Each Image and Transform BBoxes ---
        // Top-left quadrant (img 0)
        auto h0 = images[0].size(1); auto w0 = images[0].size(2);
        auto x_paste0 = std::min(center_x, (int)w0); auto y_paste0 = std::min(center_y, (int)h0);
        paste_coords[0] = {center_x - x_paste0, center_y - y_paste0};
        mosaic_image.slice(1, paste_coords[0].second, center_y).slice(2, paste_coords[0].first, center_x) =
            images[0].slice(1, h0 - y_paste0, h0).slice(2, w0 - x_paste0, w0);

        // Top-right quadrant (img 1)
        auto h1 = images[1].size(1); auto w1 = images[1].size(2);
        auto x_paste1 = std::min(output_w - center_x, (int)w1); auto y_paste1 = std::min(center_y, (int)h1);
        paste_coords[1] = {center_x, center_y - y_paste1};
        mosaic_image.slice(1, paste_coords[1].second, center_y).slice(2, center_x, center_x + x_paste1) =
            images[1].slice(1, h1 - y_paste1, h1).slice(2, 0, x_paste1);

        // Bottom-left quadrant (img 2)
        auto h2 = images[2].size(1); auto w2 = images[2].size(2);
        auto x_paste2 = std::min(center_x, (int)w2); auto y_paste2 = std::min(output_h - center_y, (int)h2);
        paste_coords[2] = {center_x - x_paste2, center_y};
        mosaic_image.slice(1, center_y, center_y + y_paste2).slice(2, paste_coords[2].first, center_x) =
            images[2].slice(1, 0, y_paste2).slice(2, w2 - x_paste2, w2);

        // Bottom-right quadrant (img 3)
        auto h3 = images[3].size(1); auto w3 = images[3].size(2);
        auto x_paste3 = std::min(output_w - center_x, (int)w3); auto y_paste3 = std::min(output_h - center_y, (int)h3);
        paste_coords[3] = {center_x, center_y};
        mosaic_image.slice(1, center_y, center_y + y_paste3).slice(2, center_x, center_x + x_paste3) =
            images[3].slice(1, 0, y_paste3).slice(2, 0, x_paste3);

        // --- Transform Bounding Boxes ---
        std::vector<torch::Tensor> combined_bboxes_list;
        for (int i = 0; i < 4; ++i) {
            if (bboxes[i].numel() == 0) continue;

            torch::Tensor new_bboxes = bboxes[i].clone();
            // Translate bboxes by the top-left coordinate where their image was pasted
            new_bboxes.select(1, 0) += paste_coords[i].first; // xmin
            new_bboxes.select(1, 2) += paste_coords[i].first; // xmax
            new_bboxes.select(1, 1) += paste_coords[i].second; // ymin
            new_bboxes.select(1, 3) += paste_coords[i].second; // ymax

            // Clip bboxes to the mosaic image dimensions
            new_bboxes.select(1, 0).clamp_(0, output_w); // xmin
            new_bboxes.select(1, 2).clamp_(0, output_w); // xmax
            new_bboxes.select(1, 1).clamp_(0, output_h); // ymin
            new_bboxes.select(1, 3).clamp_(0, output_h); // ymax

            // Filter out bboxes that are now too small
            torch::Tensor w = new_bboxes.select(1, 2) - new_bboxes.select(1, 0);
            torch::Tensor h = new_bboxes.select(1, 3) - new_bboxes.select(1, 1);
            torch::Tensor keep = (w > 2) & (h > 2); // Use a threshold > 1 for robustness

            if (keep.any().item<bool>()) {
                combined_bboxes_list.push_back(new_bboxes.index({keep}));
            }
        }

        torch::Tensor final_bboxes;
        if (!combined_bboxes_list.empty()) {
            final_bboxes = torch::cat(combined_bboxes_list, 0);
        } else {
            // Return an empty tensor of the correct shape if no bboxes remain
            final_bboxes = torch::empty({0, bboxes[0].size(1)}, bboxes[0].options());
        }

        return std::make_pair(mosaic_image, final_bboxes);
    }

} // namespace xt::transforms::image