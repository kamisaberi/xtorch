#include <transforms/image/random_order.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_order.h"
// #include "transforms/image/random_horizontal_flip.h"
// #include "transforms/image/random_gamma.h"
// #include "transforms/image/random_crop.h"
// #include "utils/image_conversion.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
//
// int main() {
//     // 1. Create a sample image.
//     torch::Tensor image = torch::zeros({3, 300, 300});
//     // Draw a large 'F' so orientation and cropping are obvious.
//     image.index_put_({torch::indexing::Slice(), torch::indexing::Slice(50, 250), torch::indexing::Slice(50, 80)}, 1.0);
//     image.index_put_({torch::indexing::Slice(), torch::indexing::Slice(50, 80), torch::indexing::Slice(80, 200)}, 1.0);
//     image.index_put_({torch::indexing::Slice(), torch::indexing::Slice(140, 170), torch::indexing::Slice(80, 180)}, 1.0);
//
//     cv::imwrite("order_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved order_before.png" << std::endl;
//
//     // 2. Create a list of transforms to apply.
//     // We use deterministic versions (p=1.0) to make the effect clear.
//     std::vector<std::shared_ptr<xt::Module>> transforms_to_apply = {
//         std::make_shared<xt::transforms::image::RandomHorizontalFlip>(1.0),
//         std::make_shared<xt::transforms::image::RandomGamma>(std::make_pair(1.8, 1.8), 1.0), // Darken
//         std::make_shared<xt::transforms::image::RandomCrop>(std::make_pair(224, 224))
//     };
//
//     std::cout << "--- Applying RandomOrder ---" << std::endl;
//
//     // 3. Create the RandomOrder transform.
//     xt::transforms::image::RandomOrder random_order_transform(transforms_to_apply);
//
//     // 4. Apply the transform.
//     // The order of Flip, Gamma, Crop will be different each time this is run.
//     torch::Tensor result_tensor = std::any_cast<torch::Tensor>(random_order_transform.forward({image}));
//
//     // 5. Save the result.
//     cv::Mat result_mat = xt::utils::image::tensor_to_mat_8u(result_tensor);
//     cv::imwrite("order_after.png", result_mat);
//     std::cout << "Saved order_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomOrder::RandomOrder() {
        std::random_device rd;
        gen_.seed(rd());
    }

    RandomOrder::RandomOrder(const std::vector<std::shared_ptr<xt::Module>>& transforms)
        : transforms_(transforms) {

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomOrder::forward(std::initializer_list<std::any> tensors) -> std::any {
        if (transforms_.empty()) {
            return tensors.begin()[0];
        }

        // --- Shuffle the order of transforms ---
        // We shuffle a vector of indices rather than the vector of shared_ptrs
        // to avoid modifying the original order stored in the object.
        std::vector<int> indices(transforms_.size());
        std::iota(indices.begin(), indices.end(), 0); // Fill with 0, 1, 2, ...
        std::shuffle(indices.begin(), indices.end(), gen_);

        // --- Apply the transforms in the new random order ---
        // Start with the initial input. Note: transforms that take multiple inputs
        // (like Mosaic) won't work correctly inside RandomOrder.
        std::any current_result = tensors.begin()[0];

        for (int index : indices) {
            current_result = transforms_[index]->forward({current_result});
        }

        return current_result;
    }

} // namespace xt::transforms::image