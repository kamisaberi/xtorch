#include <transforms/image/ada.h>


// #include "transforms/image/ada.h"
// #include "transforms/image/flip.h"      // Assume you have a HorizontalFlip transform
// #include "transforms/image/rotate.h"    // Assume you have a Rotate transform
// #include "transforms/image/oversampling.h" // Your 10-crop OverSampling transform
//
// #include <iostream>
//
// int main() {
//     // --- 1. Create instances of the simple augmentations you want to use ---
//     xt::transforms::image::HorizontalFlip flipper;
//     xt::transforms::image::Rotate rotator(15); // Rotate by 15 degrees
//
//     // Your 10-crop OverSampling is also a valid augmentation transform for ADA.
//     // It will produce 10 augmented images in one go.
//     xt::transforms::image::OverSampling ten_cropper({224, 224});
//
//     // --- 2. Create a list of pointers to these transforms ---
//     std::vector<xt::Module*> tta_transforms = {
//         &flipper,
//         &rotator,
//         &ten_cropper // This transform will add 10 images to the batch
//     };
//
//     // --- 3. Instantiate the ADA transform with this list ---
//     xt::transforms::image::ADA ada_transform(tta_transforms);
//
//     // --- 4. Apply the ADA transform to a new image ---
//     torch::Tensor input_image = torch::randn({3, 256, 256});
//
//     std::any result_any = ada_transform.forward({input_image});
//     torch::Tensor tta_batch = std::any_cast<torch::Tensor>(result_any);
//
//     // --- 5. Check the output ---
//     // Expected size: 1 (original) + 1 (flip) + 1 (rotate) + 10 (crops) = 13
//     std::cout << "Original image shape: " << input_image.sizes() << std::endl;
//     std::cout << "Generated TTA batch shape: " << tta_batch.sizes() << std::endl;
//     // Expected output: [13, 3, 256, 256] (or cropped size for the crops)
//
//     // Now you would pass `tta_batch` to your model:
//     // torch::Tensor predictions = model->forward(tta_batch);
//
//     // And then aggregate the predictions:
//     // torch::Tensor final_prediction = predictions.mean(0); // e.g., for logits
//
//     return 0;
// }


namespace xt::transforms::image {

    ADA::ADA() = default;

    ADA::ADA(std::vector<xt::Module*> augmentations)
        : augmentations_(augmentations) {}

    auto ADA::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("ADA::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to ADA is not defined.");
        }

        // 2. --- Create the batch of augmented images ---

        // The final batch will contain the original image plus all augmented versions.
        // Reserve space for efficiency.
        std::vector<torch::Tensor> augmented_batch;
        augmented_batch.reserve(augmentations_.size() + 1);

        // Add the original, untransformed image as the first element. This is standard practice.
        augmented_batch.push_back(input_tensor);

        // 3. --- Apply each augmentation transform ---
        for (xt::Module* transform : augmentations_) {
            if (transform) {
                // Call the forward method of the provided transform module.
                // We use a clone to ensure that one transform doesn't affect the input
                // for the next transform in the list.
                std::any augmented_any = transform->forward({input_tensor.clone()});

                // The result could be a single tensor or a batch of tensors (e.g. from 10-crop)
                try {
                     torch::Tensor result_tensor = std::any_cast<torch::Tensor>(augmented_any);
                     if (result_tensor.dim() == input_tensor.dim()) {
                        // It's a single augmented image, add it to the batch
                        augmented_batch.push_back(result_tensor);
                     } else if (result_tensor.dim() == input_tensor.dim() + 1) {
                        // It's already a batch of images (e.g., from OverSampling/10-crop)
                        // Unbind it into individual images and add them to our batch
                        auto unstacked = torch::unbind(result_tensor, 0);
                        augmented_batch.insert(augmented_batch.end(), unstacked.begin(), unstacked.end());
                     }
                } catch (const std::bad_any_cast& e) {
                    // Handle other possible return types if necessary
                    throw std::runtime_error("ADA received an unexpected return type from a sub-transform.");
                }
            }
        }

        // 4. --- Stack the final list of tensors into a single batch ---
        return torch::stack(augmented_batch, 0);
    }

} // namespace xt::transforms::image