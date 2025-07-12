#pragma once
#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomCrop
     * @brief Crops a given image tensor at a random location.
     *
     * This transform crops a patch of a specified `size` from the input image.
     * If the input image is smaller than the crop size, it can be padded first.
     */
    class RandomCrop : public xt::Module {
    public:
        /**
         * @brief Default constructor. Does nothing as a crop size is required.
         *        Provided for completeness but should not be used directly.
         */
        RandomCrop();

        /**
         * @brief Constructs the RandomCrop transform.
         *
         * @param size A pair `{height, width}` specifying the desired output size.
         * @param padding An integer specifying the padding to add to each border
         *                of the image before cropping. Can be used to make cropping
         *                from the edges more likely.
         * @param pad_if_needed A boolean that, if true, will pad the image to the
         *                      target size if it is smaller than the crop size.
         * @param fill The constant value used for padding.
         */
        explicit RandomCrop(
            std::pair<int, int> size,
            int padding = 0,
            bool pad_if_needed = false,
            double fill = 0.0
        );

        /**
         * @brief Executes the random crop operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting cropped torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Helper to get the top-left coordinates for the crop
        static std::pair<int, int> get_crop_params(
            const torch::Tensor& img,
            const std::pair<int, int>& output_size,
            std::mt19937& gen
        );

        std::pair<int, int> size_;
        int padding_;
        bool pad_if_needed_;
        double fill_;

        std::mt19937 gen_;
    };

} // namespace xt::transforms::image