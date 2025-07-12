#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class RandomMosaic
     * @brief Creates a mosaic image by combining four input images.
     *
     * This transform implements the Mosaic data augmentation technique. It takes
     * four images and their corresponding bounding boxes, and combines them into a
     * single canvas of a specified `output_size`. A random center point is chosen,
     * and the four images are placed into the four quadrants around this center.
     * This is a deterministic operation; the randomness is in how the mosaic
     * is constructed, not whether it is applied.
     */
    class RandomMosaic : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates a mosaic with a 640x640 output size.
         */
        RandomMosaic();

        /**
         * @brief Constructs the Mosaic transform.
         *
         * @param output_size A pair `{height, width}` for the final mosaic image.
         * @param fill The constant value used for padding areas if images don't
         *             fill the entire canvas. Value is for float tensors [0, 1].
         */
        explicit RandomMosaic(
            std::pair<int, int> output_size,
            double fill = 0.5
        );

        /**
         * @brief Executes the mosaic augmentation.
         * @param tensors An initializer list expected to contain 8 items in sequence:
         *                {img1, bboxes1, img2, bboxes2, img3, bboxes3, img4, bboxes4}
         *                - img: A 3D image tensor (C, H, W).
         *                - bboxes: A 2D tensor of shape [N, >=5] where the first 4
         *                  columns are [xmin, ymin, xmax, ymax] and the 5th is the class_id.
         * @return An std::any containing a std::pair<torch::Tensor, torch::Tensor>.
         *         - The first element is the resulting mosaic image tensor.
         *         - The second element is the combined and transformed bounding box tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::pair<int, int> output_size_;
        double fill_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image