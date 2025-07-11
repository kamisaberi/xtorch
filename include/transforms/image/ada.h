#pragma once

#include "../common.h"

#pragma once

// Forward declaration is good practice if you only need pointers/references
// namespace xt::transforms { class Module; }

namespace xt::transforms::image {

    /**
     * @class ADA
     * @brief Implements an Augment-and-Aggregate (ADA) or Test-Time Augmentation (TTA) transform.
     *
     * This transform takes a single input image and a list of other augmentation
     * transforms (e.g., flip, rotate, crop). It applies each of these transforms
     * to the original image and stacks the results into a single batch. This batch
     * can then be fed to a model to get multiple predictions, which are aggregated
     * later for a more robust final result.
     *
     * It also includes the original, untransformed image as the first element in the batch.
     */
    class ADA : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an empty ADA transform.
         */
        ADA();

        /**
         * @brief Constructs the ADA transform with a list of augmentations.
         * @param augmentations A vector of pointers to other transform modules
         *                      that will be applied to the input image.
         */
        explicit ADA(std::vector<xt::Module*> augmentations);

        /**
         * @brief Applies the list of augmentations to the input image.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting stacked torch::Tensor of
         *         shape [num_augmentations + 1, C, H, W].
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Stores the list of augmentation transforms to apply.
        // Using pointers allows for polymorphism (e.g., a vector containing
        // both Flip and Rotate objects).
        std::vector<xt::Module*> augmentations_;
    };

} // namespace xt::transforms::image